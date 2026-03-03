"""
Run inference on a .pte file exported via ExecuTorch with the XNNPACK backend.

Usage:
    python run_inference_pte.py --model gemma-3-1b-it_int8.pte --prompt "Hello, how are you?"
    python run_inference_pte.py --model gemma-3-1b-it_int8.pte --prompt "Explain gravity" --max_tokens 128
"""

import argparse
import ctypes
import glob
import importlib
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Register ExecuTorch operator kernels (must happen BEFORE loading .pte).
#
# The XNNPACK delegate handles most delegated ops, but fallback ops like
# quantized_decomposed::embedding_byte run on CPU kernels that must be
# explicitly loaded.  We try multiple strategies to ensure the kernel is
# available.
# ---------------------------------------------------------------------------

def _try_import(module_path: str) -> bool:
    try:
        importlib.import_module(module_path)
        print(f"  Loaded: {module_path}")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _try_load_so(pattern_fragments: list[str]) -> bool:
    """Search site-packages for a .so matching the pattern and dlopen it."""
    for sp in sys.path:
        if not os.path.isdir(sp):
            continue
        for frag in pattern_fragments:
            matches = glob.glob(os.path.join(sp, "**", frag), recursive=True)
            for so_path in matches:
                try:
                    ctypes.CDLL(so_path)
                    print(f"  Loaded .so: {so_path}")
                    return True
                except OSError:
                    continue
    return False


# Strategy 1: import Python kernel modules (registers via side-effect)
for mod in [
    "executorch.kernels.portable",
    "executorch.kernels.quantized",
    "executorch.extension.llm.custom_ops",
]:
    _try_import(mod)

# Strategy 2: if the quantized kernel still isn't importable via Python,
# try to find and dlopen the shared library directly.
_try_load_so([
    "libquantized_ops_aot_lib.*so*",
    "libquantized_kernels.*so*",
    "quantized_ops_aot_lib.*so*",
    "quantized.*cpython*.so",
])

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Model loading — try the high-level Runtime API first, fall back to the
# lower-level _load_for_executorch pybinding which bundles its own kernel
# registry and is often more complete.
# ---------------------------------------------------------------------------

_load_for_executorch = None
try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch as _lfe,
    )
    _load_for_executorch = _lfe
    print("  Available: _load_for_executorch (portable_lib)")
except ImportError:
    pass


def load_pte(model_path: str):
    """Load a .pte program and return a callable that wraps .forward()."""

    # --- Try high-level Runtime API first ---
    try:
        from executorch.runtime import Runtime
        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        method = program.load_method("forward")
        print("  Loaded via Runtime API")
        return method
    except Exception as e:
        print(f"  Runtime API failed ({e}), trying portable_lib fallback...")

    # --- Fallback: portable_lib._load_for_executorch ---
    if _load_for_executorch is not None:
        module = _load_for_executorch(model_path)
        print("  Loaded via _load_for_executorch")
        return module

    raise RuntimeError(
        f"Could not load {model_path}. "
        "Neither Runtime API nor _load_for_executorch succeeded."
    )


def run_forward(model, input_ids, attention_mask):
    """Execute forward pass, handling both Runtime Method and portable Module."""
    # Runtime Method API: method.execute([...])
    if hasattr(model, "execute"):
        return model.execute([input_ids, attention_mask])
    # portable_lib Module API: module.forward((..,...))
    if hasattr(model, "forward"):
        return model.forward((input_ids, attention_mask))
    raise RuntimeError("Model object has neither .execute() nor .forward()")


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
    max_seq_len: int = 512,
) -> str:
    """Autoregressive token generation using the ExecuTorch model."""

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
            generated_ids = generated_ids[:, -max_seq_len:]
            seq_len = max_seq_len

        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        outputs = run_forward(model, generated_ids, attention_mask)

        # outputs may be a list/tuple of tensors
        logits = outputs[0]  # shape: [1, seq_len, vocab_size]

        next_logits = logits[:, -1, :].float()

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float("-inf")

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

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading .pte model: {args.model}")
    model = load_pte(args.model)
    print("Model loaded.\n")

    print(f"Prompt: {args.prompt}\n")
    output = generate(
        model=model,
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
