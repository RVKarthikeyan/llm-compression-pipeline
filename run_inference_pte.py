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
        pte_files.sort(key=lambda f: os.path.getsize(os.path.join(model_arg, f)), reverse=True)
        return os.path.join(model_arg, pte_files[0])
    raise FileNotFoundError(f"Not a valid file or directory: {model_arg}")


def sample_next_token(logits, temperature, top_k):
    """Sample a single next token from logits [1, 1, vocab_size]."""
    next_logits = logits[:, -1, :].float()

    if temperature > 0:
        next_logits = next_logits / temperature
    if top_k > 0:
        topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
        next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")
    if temperature > 0:
        probs = torch.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    else:
        return torch.argmax(next_logits, dim=-1, keepdim=True)


def generate(model, tokenizer, prompt, max_new_tokens=64,
             temperature=0.7, top_k=50, max_seq_len=512):
    """
    Token-by-token generation for static-shape ExecuTorch models.

    The XNNPACK recipe from optimum-cli exports models with:
      - Static input shape (1, 1): one token per forward call
      - Internal KV-cache as mutable buffers (state persists between calls)
      - Inputs: (input_ids [1,1], cache_position [1])
    """
    token_ids = tokenizer.encode(prompt)
    prompt_len = len(token_ids)
    if prompt_len >= max_seq_len:
        print(f"Warning: truncating prompt to {max_seq_len} tokens.")
        token_ids = token_ids[:max_seq_len]
        prompt_len = max_seq_len

    print(f"Prompt tokens: {prompt_len}")

    # ------------------------------------------------------------------
    # Prefill: feed each prompt token one at a time to fill the KV-cache
    # ------------------------------------------------------------------
    print("Prefilling...", end="", flush=True)
    t_start = time.perf_counter()

    for i, tok in enumerate(token_ids):
        token_tensor = torch.tensor([[tok]], dtype=torch.long)       # (1, 1)
        pos_tensor = torch.tensor([i], dtype=torch.long)             # (1,)
        logits = model.forward((token_tensor, pos_tensor))

    t_prefill = time.perf_counter() - t_start
    print(f" done ({t_prefill:.2f}s, {prompt_len / t_prefill:.1f} tok/s)")

    # ------------------------------------------------------------------
    # Decode: sample from last logits, then feed new tokens one at a time
    # ------------------------------------------------------------------
    print("Generating", end="", flush=True)
    t_decode_start = time.perf_counter()

    generated_tokens = []
    logits_out = logits[0] if isinstance(logits, (list, tuple)) else logits

    for step in range(max_new_tokens):
        next_token = sample_next_token(logits_out, temperature, top_k)
        tok_id = next_token.item()

        if tok_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(tok_id)
        print(".", end="", flush=True)

        pos = prompt_len + step
        if pos >= max_seq_len:
            print("\n  Reached max_seq_len, stopping.")
            break

        token_tensor = torch.tensor([[tok_id]], dtype=torch.long)    # (1, 1)
        pos_tensor = torch.tensor([pos], dtype=torch.long)           # (1,)
        logits = model.forward((token_tensor, pos_tensor))
        logits_out = logits[0] if isinstance(logits, (list, tuple)) else logits

    t_decode = time.perf_counter() - t_decode_start
    n_gen = len(generated_tokens)
    print()
    print(f"Generated {n_gen} tokens in {t_decode:.2f}s "
          f"({n_gen / t_decode:.1f} tok/s)" if t_decode > 0 else
          f"Generated {n_gen} tokens")

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


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
