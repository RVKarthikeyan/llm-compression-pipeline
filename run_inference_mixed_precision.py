"""
Run inference on a .pte file exported via mixed_precision_quantization_executorch.py.

This model uses:
  - Dynamic input shape (1, seq_len) with seq_len in [1, 512]
  - No internal KV-cache (full-sequence forward pass each time)
  - Inputs: (input_ids [1, seq_len], attention_mask [1, seq_len])
  - Output: logits [1, seq_len, vocab_size]

Usage:
    python run_inference_mixed_precision.py --model gemma-3-1b-it_int8.pte --prompt "Hello"
    python run_inference_mixed_precision.py --model gemma-3-1b-it_int8.pte --prompt "Explain gravity" --max_tokens 128
"""

import argparse
import importlib
import os
import time

import torch


def register_kernels():
    """Register ExecuTorch kernels needed for XNNPACK + quantized ops."""
    modules = [
        "executorch.kernels.portable",
        "executorch.kernels.quantized",
        "executorch.extension.llm.custom_ops",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"  Registered: {mod}")
        except (ImportError, ModuleNotFoundError):
            pass


def load_model(pte_path: str):
    """Load a .pte file using the ExecuTorch portable runtime."""
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    model = _load_for_executorch(pte_path)
    print(f"  Loaded via _load_for_executorch")
    return model


def resolve_pte_path(model_arg: str) -> str:
    """Accept a .pte file path or a directory containing .pte files."""
    if os.path.isfile(model_arg):
        return model_arg
    if os.path.isdir(model_arg):
        pte_files = [f for f in os.listdir(model_arg) if f.endswith(".pte")]
        if not pte_files:
            raise FileNotFoundError(f"No .pte files in {model_arg}")
        pte_files.sort(
            key=lambda f: os.path.getsize(os.path.join(model_arg, f)), reverse=True
        )
        return os.path.join(model_arg, pte_files[0])
    raise FileNotFoundError(f"Not a valid file or directory: {model_arg}")


def sample_next_token(logits, temperature, top_k):
    """Sample a single next token from logits at the last position."""
    next_logits = logits[:, -1, :].float()

    if temperature > 0:
        next_logits = next_logits / temperature
    if top_k > 0:
        topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
        next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")
    if temperature > 0:
        probs = torch.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    else:
        return torch.argmax(next_logits, dim=-1).item()


def generate(model, tokenizer, prompt, max_new_tokens=64,
             temperature=0.7, top_k=50, max_seq_len=512):
    """
    Autoregressive generation for a no-KV-cache ExecuTorch model.

    Since the model has no internal KV-cache, each forward pass processes
    the full sequence from scratch. This is slower than cached generation
    but matches how mixed_precision_quantization_executorch.py exports
    the model (use_cache=False).
    """
    token_ids = tokenizer.encode(prompt)
    prompt_len = len(token_ids)

    if prompt_len >= max_seq_len:
        print(f"Warning: truncating prompt to {max_seq_len - 1} tokens "
              f"(need room for generation).")
        token_ids = token_ids[:max_seq_len - 1]
        prompt_len = len(token_ids)

    print(f"Prompt tokens: {prompt_len}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Max seq length: {max_seq_len}\n")

    generated_tokens = []
    total_seq = list(token_ids)

    # --- First forward pass with the full prompt ---
    print("Running prompt forward pass...", end="", flush=True)
    t_start = time.perf_counter()

    input_ids = torch.tensor([total_seq], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    logits = model.forward((input_ids, attention_mask))
    logits_out = logits[0] if isinstance(logits, (list, tuple)) else logits

    t_prefill = time.perf_counter() - t_start
    print(f" done ({t_prefill:.2f}s)")

    # Sample first new token
    tok_id = sample_next_token(logits_out, temperature, top_k)

    if tok_id == tokenizer.eos_token_id:
        print("Model produced EOS immediately.")
        return ""

    generated_tokens.append(tok_id)
    total_seq.append(tok_id)

    # Stream the first decoded token
    print(f"\n--- Output ---")
    print(tokenizer.decode(generated_tokens, skip_special_tokens=True), end="", flush=True)

    # --- Autoregressive loop ---
    t_decode_start = time.perf_counter()

    for step in range(1, max_new_tokens):
        if len(total_seq) >= max_seq_len:
            print(f"\nReached max_seq_len ({max_seq_len}), stopping.")
            break

        input_ids = torch.tensor([total_seq], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        logits = model.forward((input_ids, attention_mask))
        logits_out = logits[0] if isinstance(logits, (list, tuple)) else logits

        tok_id = sample_next_token(logits_out, temperature, top_k)

        if tok_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(tok_id)
        total_seq.append(tok_id)

        # Stream decoded output so far
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\r--- Output ---\n{decoded}", end="", flush=True)

    t_decode = time.perf_counter() - t_decode_start
    n_gen = len(generated_tokens)
    t_total = time.perf_counter() - t_start
    print()

    # --- Stats ---
    print(f"\n{'='*60}")
    print(f"  Prompt tokens   : {prompt_len}")
    print(f"  Generated tokens: {n_gen}")
    print(f"  Prefill time    : {t_prefill:.2f}s")
    if n_gen > 1:
        print(f"  Decode time     : {t_decode:.2f}s ({(n_gen - 1) / t_decode:.2f} tok/s)")
    print(f"  Total time      : {t_total:.2f}s")
    print(f"{'='*60}")

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a mixed-precision .pte model (ExecuTorch, no KV-cache)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pte file or directory containing one")
    parser.add_argument("--tokenizer", type=str, default="google/gemma-3-1b-it",
                        help="HuggingFace tokenizer name or local path")
    parser.add_argument("--prompt", type=str,
                        default="What is the capital of France?")
    parser.add_argument("--max_tokens", type=int, default=64,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum total sequence length (must be <= 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    args = parser.parse_args()

    if args.max_seq_len > 512:
        print("Warning: model was exported with max seq_len=512, clamping.")
        args.max_seq_len = 512

    # --- Register kernels ---
    print("Registering ExecuTorch kernels...")
    register_kernels()

    # --- Tokenizer ---
    print(f"\nLoading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- Load model ---
    pte_path = resolve_pte_path(args.model)
    print(f"\nLoading model: {pte_path}")
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")
    model = load_model(pte_path)
    print("Model loaded.\n")

    # --- Generate ---
    print(f"Prompt: {args.prompt}\n")
    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        max_seq_len=args.max_seq_len,
    )

    print(f"\nFinal output:\n{output}")


if __name__ == "__main__":
    main()
