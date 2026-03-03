"""
Run inference on a .pte file exported via optimum-cli export executorch.

Usage:
    python run_inference_pte.py --model ./model.pte --prompt "Hello, how are you?"
    python run_inference_pte.py --model ./model.pte --prompt "Explain gravity" --max_tokens 128

The model should have been exported with:
    optimum-cli export executorch \
        --model "google/gemma-3-1b-it" \
        --task "text-generation" \
        --recipe "xnnpack" \
        --output_dir "./model.pte"
"""

import argparse
import time

import torch
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM


def generate(
    model: ExecuTorchModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate text using the ExecuTorch model via optimum's generate()."""

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    print(f"Prompt tokens: {prompt_len}")
    print("Generating...")

    t_start = time.perf_counter()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    t_elapsed = time.perf_counter() - t_start
    n_generated = generated_ids.shape[1] - prompt_len

    print(f"Generated {n_generated} tokens in {t_elapsed:.2f}s "
          f"({n_generated / t_elapsed:.1f} tok/s)")

    output_text = tokenizer.decode(
        generated_ids[0, prompt_len:], skip_special_tokens=True
    )
    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a .pte model exported with optimum-cli + ExecuTorch XNNPACK"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the .pte output directory (from optimum-cli export executorch --output_dir)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="HuggingFace tokenizer name or path (default: same as --model, or google/gemma-3-1b-it)"
    )
    parser.add_argument(
        "--prompt", type=str, default="What is the capital of France?",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=64,
        help="Maximum number of new tokens to generate (default: 64)"
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        # If the output_dir doesn't contain tokenizer files, fall back to the
        # original model name.
        fallback = "google/gemma-3-1b-it"
        print(f"  Tokenizer not found at {tokenizer_path}, falling back to {fallback}")
        tokenizer = AutoTokenizer.from_pretrained(fallback)

    print(f"Loading ExecuTorch model: {args.model}")
    model = ExecuTorchModelForCausalLM.from_pretrained(args.model)
    print("Model loaded.\n")

    print(f"Prompt: {args.prompt}\n")
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
    )

    print(f"\n{'='*60}")
    print(f"Output:\n{output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
