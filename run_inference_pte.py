"""
Run inference on a .pte file exported via ExecuTorch with the XNNPACK backend.

Usage:
    python run_inference_pte.py --model gemma-3-1b-it_int8.pte --prompt "Hello, how are you?"
    python run_inference_pte.py --model gemma-3-1b-it_int8.pte --prompt "Explain gravity" --max_tokens 128
"""

import argparse
import importlib
import time

import torch

# ---------------------------------------------------------------------------
# Register ExecuTorch operator kernels (must happen BEFORE loading .pte).
#
# The XNNPACK delegate handles most ops, but fallback ops like
# quantized_decomposed::embedding_byte run on the portable/quantized
# CPU kernels.  Importing these modules triggers kernel registration.
# ---------------------------------------------------------------------------
_KERNEL_LIBS = [
    "executorch.kernels.portable",   # core portable ops
    "executorch.kernels.quantized",  # quantized ops (embedding_byte, etc.)
]
for _lib in _KERNEL_LIBS:
    try:
        importlib.import_module(_lib)
        print(f"  Registered kernels: {_lib}")
    except ImportError:
        print(f"  Warning: could not import {_lib} — some ops may be unavailable")

from executorch.runtime import Runtime, Program, Method
from transformers import AutoTokenizer


def load_pte(model_path: str) -> Method:
    """Load a .pte program and return the 'forward' method."""
    runtime = Runtime.get()
    program = runtime.load_program(model_path)
    method = program.load_method("forward")
    return method


def generate(
    method: Method,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
    max_seq_len: int = 512,
) -> str:
    """Autoregressive token generation using the ExecuTorch method."""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.long)
    prompt_len = input_ids.shape[1]

    if prompt_len >= max_seq_len:
        print(f"Warning: prompt length ({prompt_len}) >= max_seq_len ({max_seq_len}), truncating.")
        input_ids = input_ids[:, :max_seq_len]
        prompt_len = max_seq_len

    generated_ids = input_ids.clone()

    print(f"Prompt tokens: {prompt_len}")
    print("Generating", end="", flush=True)

    t_start = time.perf_counter()

    for step in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        if seq_len > max_seq_len:
            # Slide window to stay within the model's supported range
            generated_ids = generated_ids[:, -max_seq_len:]
            seq_len = max_seq_len

        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        # Run the ExecuTorch forward pass
        outputs = method.execute([generated_ids, attention_mask])
        logits = outputs[0]  # shape: [1, seq_len, vocab_size]

        # Take logits for the last position
        next_logits = logits[:, -1, :].float()

        # Apply temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float("-inf")

        # Sample or greedy
        if temperature > 0:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        print(".", end="", flush=True)

    t_elapsed = time.perf_counter() - t_start
    n_generated = generated_ids.shape[1] - prompt_len

    print()
    print(f"Generated {n_generated} tokens in {t_elapsed:.2f}s "
          f"({n_generated / t_elapsed:.1f} tok/s)")

    # Decode only the newly generated tokens
    output_text = tokenizer.decode(
        generated_ids[0, prompt_len:], skip_special_tokens=True
    )
    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a .pte model exported with ExecuTorch + XNNPACK"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the .pte file"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="google/gemma-3-1b-it",
        help="HuggingFace tokenizer name or path (default: google/gemma-3-1b-it)"
    )
    parser.add_argument(
        "--prompt", type=str, default="What is the capital of France?",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=64,
        help="Maximum number of new tokens to generate (default: 64)"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512,
        help="Maximum total sequence length supported by the model (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature; 0 for greedy (default: 0.7)"
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling; 0 to disable (default: 50)"
    )
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load .pte model
    print(f"Loading .pte model: {args.model}")
    method = load_pte(args.model)
    print("Model loaded.\n")

    # Run generation
    print(f"Prompt: {args.prompt}\n")
    output = generate(
        method=method,
        tokenizer=tokenizer,
        prompt=args.prompt,
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
