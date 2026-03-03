"""
Run inference on a .pte file exported via ExecuTorch with XNNPACK.

Usage:
    python run_inference_pte.py --model gemma3_8da4w.pte --prompt "Hello"
    python run_inference_pte.py --model ./model_dir/ --prompt "Hello" --max_tokens 128

Exported with:
    optimum-cli export executorch \
        --model "google/gemma-3-1b-it" \
        --task "text-generation" \
        --recipe "xnnpack" \
        --qlinear 8da4w \
        --output_dir "./model_dir"
"""

import argparse
import importlib
import os
import time

import torch

# ---------------------------------------------------------------------------
# Register ExecuTorch kernels BEFORE loading any .pte file.
# ---------------------------------------------------------------------------
for _mod in [
    "executorch.kernels.portable",
    "executorch.kernels.quantized",
    "executorch.extension.llm.custom_ops",
]:
    try:
        importlib.import_module(_mod)
    except (ImportError, ModuleNotFoundError):
        pass

from executorch.extension.pybindings.portable_lib import _load_for_executorch
from transformers import AutoTokenizer


def resolve_pte_path(model_arg: str) -> str:
    """Accept a .pte file path or a directory containing .pte files."""
    if os.path.isfile(model_arg):
        return model_arg
    if os.path.isdir(model_arg):
        pte_files = [f for f in os.listdir(model_arg) if f.endswith(".pte")]
        if not pte_files:
            raise FileNotFoundError(f"No .pte files in {model_arg}")
        # Pick the largest file (most likely the full model)
        pte_files.sort(key=lambda f: os.path.getsize(os.path.join(model_arg, f)), reverse=True)
        return os.path.join(model_arg, pte_files[0])
    raise FileNotFoundError(f"Not a valid file or directory: {model_arg}")


def generate(model, tokenizer, prompt, max_new_tokens=64,
             temperature=0.7, top_k=50, max_seq_len=512):
    """Autoregressive token-by-token generation."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.long)
    prompt_len = input_ids.shape[1]
    if prompt_len >= max_seq_len:
        print(f"Warning: truncating prompt to {max_seq_len} tokens.")
        input_ids = input_ids[:, :max_seq_len]
        prompt_len = max_seq_len

    generated_ids = input_ids.clone()

    print(f"Prompt tokens: {prompt_len}")
    print("Generating", end="", flush=True)
    t_start = time.perf_counter()

    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        if seq_len > max_seq_len:
            generated_ids = generated_ids[:, -max_seq_len:]
            seq_len = max_seq_len

        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        outputs = model.forward((generated_ids, attention_mask))
        logits = outputs[0]  # [1, seq_len, vocab_size]
        next_logits = logits[:, -1, :].float()

        if temperature > 0:
            next_logits = next_logits / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
            next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")
        if temperature > 0:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        print(".", end="", flush=True)

    t_elapsed = time.perf_counter() - t_start
    n_generated = generated_ids.shape[1] - prompt_len
    print()
    print(f"Generated {n_generated} tokens in {t_elapsed:.2f}s "
          f"({n_generated / t_elapsed:.1f} tok/s)")

    return tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a .pte model (ExecuTorch + XNNPACK)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pte file or directory containing one")
    parser.add_argument("--tokenizer", type=str, default="google/gemma-3-1b-it",
                        help="HuggingFace tokenizer name or local path")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    # --- Tokenizer ---------------------------------------------------------
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- Model (direct local load, no HuggingFace calls) -------------------
    pte_path = resolve_pte_path(args.model)
    print(f"Loading model: {pte_path}")
    model = _load_for_executorch(pte_path)
    print("Model loaded.\n")

    # --- Generate ----------------------------------------------------------
    print(f"Prompt: {args.prompt}\n")
    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        max_seq_len=args.max_seq_len,
    )

    print(f"\n{'='*60}")
    print(f"Output:\n{output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
