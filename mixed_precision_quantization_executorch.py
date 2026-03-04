import importlib, os, re, gc
import torch
import torch.nn.functional as F
from torch.export import export, Dim
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: symbol resolution
# ═════════════════════════════════════════════════════════════════════════════

def _find(symbol: str, candidates: list[str], verbose: bool = True):
    """Return the first matching object from candidate module paths, or None."""
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, symbol, None)
            if obj is not None:
                if verbose:
                    print(f"  found {symbol:40s} <- {mod_path}")
                return obj
        except Exception:
            pass
    if verbose:
        print(f"  missing {symbol:40s}   (not found)")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# NATIVE INT8 MODULE REPLACEMENT
# ═════════════════════════════════════════════════════════════════════════════

class Int8Linear(torch.nn.Module):
    """nn.Linear replacement storing weights as int8."""
    def __init__(self, weight_int8, scale, bias):
        super().__init__()
        self.register_buffer("weight_q",     weight_int8)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("linear_bias",  bias)

    def forward(self, x):
        w = self.weight_q.to(self.weight_scale.dtype) * self.weight_scale
        return F.linear(x, w, self.linear_bias)


class Int8Embedding(torch.nn.Module):
    """nn.Embedding replacement storing weights as int8."""
    def __init__(self, weight_int8, scale, padding_idx):
        super().__init__()
        self.register_buffer("weight_q",     weight_int8)
        self.register_buffer("weight_scale", scale)
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        w = self.weight_q.to(self.weight_scale.dtype) * self.weight_scale
        return F.embedding(input_ids, w, self.padding_idx)


def _quantize_weight(w_fp):
    """Symmetric per-channel quantization: fp -> int8 + scale."""
    w = w_fp.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    wq = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return wq, scale


def replace_with_int8(model, dtype):
    """Swap all Linear & Embedding layers for native-int8 versions."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "lm_head" in name:
                print(f"  Skipping {name} (keeping FP16)")
                continue
            wq, sc = _quantize_weight(module.weight.data)
            bias = module.bias.data.to(dtype) if module.bias is not None else None
            replacements.append((name, Int8Linear(wq, sc.to(dtype), bias), "linear"))
        elif isinstance(module, torch.nn.Embedding):
            print(f"  Skipping {name} (keeping FP16)")
            continue

    for name, new_mod, _ in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    embeds  = [(n, m) for n, m, t in replacements if t == "embed"]
    linears = [(n, m) for n, m, t in replacements if t == "linear"]
    tied = 0
    for en, em in embeds:
        for ln, lm in linears:
            if lm.weight_q.shape == em.weight_q.shape and torch.equal(lm.weight_q, em.weight_q):
                lm.weight_q = em.weight_q
                lm.weight_scale = em.weight_scale
                tied += 1
                print(f"  Tied: {ln} <-> {en}")

    n_lin = sum(1 for _, _, t in replacements if t == "linear")
    n_emb = sum(1 for _, _, t in replacements if t == "embed")
    return n_lin, n_emb, tied


class GemmaExportWrapper(torch.nn.Module):
    """Tensor-in / tensor-out wrapper for ExecuTorch export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXPORTED FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def run_mixed_precision_quantization(
    model_path: str = "google/gemma-3-1b-it",
    hf_token: str | None = None,
    export_dtype: torch.dtype = torch.float16,
    max_seq_len: int = 512,
    calibration_seq_len: int = 128,
    output_dir: str = ".",
    output_filename: str | None = None,
) -> dict:
    """
    End-to-end mixed-precision INT8 quantization + ExecuTorch .pte export.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local path.
    hf_token : str | None
        HuggingFace token. Falls back to ``HF_TOKEN`` env var.
    export_dtype : torch.dtype
        Dtype for scales / norms / biases (default fp16).
    max_seq_len : int
        Maximum sequence length for the dynamic shape.
    calibration_seq_len : int
        Sequence length used for the dummy calibration input.
    output_dir : str
        Directory to write the .pte file into.
    output_filename : str | None
        Override output filename. Auto-generated if None.

    Returns
    -------
    dict with keys: ``output_path``, ``size_mb``, ``quantization_path``,
    ``num_linear_replaced``, ``num_embed_replaced``.
    """

    print(f"torch {torch.__version__}\n")

    # ── Resolve imports ──────────────────────────────────────────────────────
    prepare_pt2e = _find("prepare_pt2e", [
        "torch.ao.quantization.quantize_pt2e",
        "torchao.quantization.pt2e.quantize_pt2e",
        "torch.ao.quantization._pt2e.quantize_pt2e",
        "torch.ao.quantization.pt2e",
    ])
    convert_pt2e = _find("convert_pt2e", [
        "torch.ao.quantization.quantize_pt2e",
        "torchao.quantization.pt2e.quantize_pt2e",
        "torch.ao.quantization._pt2e.quantize_pt2e",
        "torch.ao.quantization.pt2e",
    ])
    XNNPACKQuantizer = _find("XNNPACKQuantizer", [
        "executorch.backends.xnnpack.quantization.xnnpack_quantizer",
        "torchao.quantization.pt2e.xnnpack_quantizer",
        "torchao.quantization.xnnpack_quantizer",
        "torchao._executorch.xnnpack_quantizer",
        "torch.ao.quantization.quantizer.xnnpack_quantizer",
    ])
    get_sym_config = _find("get_symmetric_quantization_config", [
        "executorch.backends.xnnpack.quantization.xnnpack_quantizer",
        "torchao.quantization.pt2e.xnnpack_quantizer",
        "torchao.quantization.xnnpack_quantizer",
        "torchao._executorch.xnnpack_quantizer",
        "torch.ao.quantization.quantizer.xnnpack_quantizer",
    ])

    XnnpackPartitioner = _find("XnnpackPartitioner", [
        "executorch.backends.xnnpack.partition.xnnpack_partitioner",
    ])
    to_edge_transform_and_lower = _find("to_edge_transform_and_lower", [
        "executorch.exir",
    ])
    EdgeCompileConfig = _find("EdgeCompileConfig", [
        "executorch.exir",
    ])

    assert XnnpackPartitioner,          "XnnpackPartitioner not found"
    assert to_edge_transform_and_lower, "to_edge_transform_and_lower not found"
    assert EdgeCompileConfig,           "EdgeCompileConfig not found"

    USE_PT2E = all([prepare_pt2e, convert_pt2e, XNNPACKQuantizer, get_sym_config])
    USE_TORCHAO_QUANTIZE = False
    if not USE_PT2E:
        try:
            from torchao.quantization import quantize_ as _tq, int8_weight_only as _i8  # noqa: F401
            USE_TORCHAO_QUANTIZE = True
        except ImportError:
            pass

    path_label = (
        "PT2E + XNNPACKQuantizer"        if USE_PT2E else
        "Native INT8 module replacement" if USE_TORCHAO_QUANTIZE else
        "None (FP32 export)"
    )
    print(f"\n  Quantization path: {path_label}")

    # ── HF login ─────────────────────────────────────────────────────────────
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError(
            "HF token is empty.\n"
            "  Pass hf_token= or set the HF_TOKEN environment variable.\n"
        )
    login(token=token, add_to_git_credential=False)

    # ── Load model + tokenizer ───────────────────────────────────────────────
    is_local = os.path.isdir(model_path)
    print(f"Model : {model_path}  ({'local' if is_local else 'HuggingFace Hub'})")
    print(f"Dtype : {export_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=token if not is_local else None,
    )

    print("\nLoading model weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token if not is_local else None,
        device_map="cpu",
        torch_dtype=export_dtype,
        use_cache=False,
        attn_implementation="eager",
    )
    base_model.eval()

    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters   : {n_params / 1e6:.1f} M")
    print(f"  Linear layers: {sum(1 for m in base_model.modules() if isinstance(m, torch.nn.Linear))}")

    # ── INT8 replacement ─────────────────────────────────────────────────────
    n_lin = n_emb = n_tied = 0
    if USE_TORCHAO_QUANTIZE or not USE_PT2E:
        n_lin, n_emb, n_tied = replace_with_int8(base_model, export_dtype)

        int8_bytes, scale_bytes, other_bytes, seen = 0, 0, 0, set()
        for _, mod in base_model.named_modules():
            if isinstance(mod, (Int8Linear, Int8Embedding)):
                ptr = mod.weight_q.data_ptr()
                if ptr not in seen:
                    int8_bytes  += mod.weight_q.numel()
                    scale_bytes += mod.weight_scale.numel() * mod.weight_scale.element_size()
                    seen.add(ptr)
                if isinstance(mod, Int8Linear) and mod.linear_bias is not None:
                    other_bytes += mod.linear_bias.numel() * mod.linear_bias.element_size()
        for p in base_model.parameters():
            other_bytes += p.numel() * p.element_size()

        total = int8_bytes + scale_bytes + other_bytes
        print(f"\n  Replaced {n_lin} Linear + {n_emb} Embedding -> INT8")
        print(f"  Weight ties preserved: {n_tied}")
        print(f"  Estimated .pte size : ~{total / 1e9:.2f} GB")

    wrapped_model = GemmaExportWrapper(base_model)
    print("\nModel loaded, quantized (INT8), and wrapped.")

    # ── Trace + Lower + Export ───────────────────────────────────────────────
    print("Trace -> Lower -> Save .pte\n")

    dummy_ids  = torch.randint(0, tokenizer.vocab_size, (1, calibration_seq_len), dtype=torch.long)
    dummy_mask = torch.ones(1, calibration_seq_len, dtype=torch.long)
    example_args = (dummy_ids, dummy_mask)

    seq = Dim("seq_len", min=1, max=max_seq_len)
    dyn_shapes = ({1: seq}, {1: seq})

    print("  Tracing model graph (high RAM -- several minutes)...")
    exported_model = export(wrapped_model, example_args, dynamic_shapes=dyn_shapes)
    print("  Traced.")

    if USE_PT2E:
        quantizer = XNNPACKQuantizer()
        quant_config = get_sym_config(is_per_channel=True, is_dynamic=False)
        quantizer.set_global(quant_config)
        prepared = prepare_pt2e(exported_model, quantizer)
        exported_model = convert_pt2e(prepared)
        print("  PT2E quantization applied.")
    else:
        print("  Weights already INT8 -- skipping post-trace quantization.")

    gc.collect()

    print("  Lowering to Edge IR + XNNPACK backend...")
    executorch_program = to_edge_transform_and_lower(
        exported_model,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()

    # ── Save .pte ────────────────────────────────────────────────────────────
    if output_filename is None:
        _slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_path.rstrip("/").split("/")[-1])
        _dtype_tag = "int8" if (USE_PT2E or USE_TORCHAO_QUANTIZE or not USE_PT2E) else str(export_dtype).split(".")[-1]
        output_filename = f"{_slug}_{_dtype_tag}.pte"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  SUCCESS")
    print(f"  Model  : {model_path}")
    print(f"  File   : {os.path.abspath(output_path)}")
    print(f"  Size   : {size_mb:.2f} MB  ({size_mb/1024:.2f} GB)")
    print(f"  Storage: INT8 weights (1 byte) + {export_dtype} norms/scales")
    print(f"{'='*60}")

    print(f"\nAndroid deployment:")
    print(f"  1. Copy {output_filename} to your app's assets/ folder")
    print(f"  2. Load:  Module module = Module.load(assetFilePath(\"{output_filename}\"));")
    print(f"  3. Run :  module.forward(inputIds, attentionMask);")

    return {
        "output_path": os.path.abspath(output_path),
        "size_mb": size_mb,
        "quantization_path": path_label,
        "num_linear_replaced": n_lin,
        "num_embed_replaced": n_emb,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_mixed_precision_quantization()
