"""
Mixed-precision INT8 quantization + ExecuTorch export for Gemma-3-1B-IT.

Supports three backend recipes:
  --recipe xnnpack   CPU via XNNPACK  (default, same as original script)
  --recipe cuda      GPU via AOTInductor / CUDA backend
  --recipe vulkan    GPU via Vulkan compute shaders (Android / desktop GPU)

Usage:
    export HF_TOKEN='hf_...'
    python mixed_precision_quantization_executorch_multibackend.py --recipe xnnpack
    python mixed_precision_quantization_executorch_multibackend.py --recipe cuda
    python mixed_precision_quantization_executorch_multibackend.py --recipe vulkan
"""

import argparse
import importlib
import os
import re
import gc

import torch
import torch.nn.functional as F
from torch.export import export, Dim
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="Mixed-precision INT8 quantization + ExecuTorch export"
)
parser.add_argument(
    "--recipe", type=str, default="xnnpack",
    choices=["xnnpack", "cuda", "vulkan"],
    help="Backend recipe: xnnpack (CPU), cuda (GPU), vulkan (GPU/Android)",
)
parser.add_argument(
    "--model", type=str, default="google/gemma-3-1b-it",
    help="HuggingFace model ID or local path",
)
parser.add_argument(
    "--output", type=str, default=None,
    help="Output .pte filename (auto-generated if omitted)",
)
parser.add_argument(
    "--max_seq_len", type=int, default=512,
    help="Maximum sequence length for dynamic shape export",
)
args = parser.parse_args()

RECIPE     = args.recipe
MODEL_PATH = args.model
MAX_SEQ    = args.max_seq_len

print(f"torch {torch.__version__}")
print(f"Recipe: {RECIPE}\n")

# ═════════════════════════════════════════════════════════════════════════════
# IMPORT RESOLUTION — locate PT2E / backend / ExecuTorch symbols
# ═════════════════════════════════════════════════════════════════════════════

def _find(symbol: str, candidates: list[str]):
    """Return the first matching object from candidate module paths, or None."""
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, symbol, None)
            if obj is not None:
                print(f"  found {symbol:40s} <- {mod_path}")
                return obj
        except Exception:
            pass
    print(f"  missing {symbol:40s}   (not found)")
    return None

# ── PT2E quantization (optional — only used with xnnpack PT2E path) ───────
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

# ── ExecuTorch core lowering (required for all recipes) ──────────────────
to_edge_transform_and_lower = _find("to_edge_transform_and_lower", [
    "executorch.exir",
])
EdgeCompileConfig = _find("EdgeCompileConfig", [
    "executorch.exir",
])

assert to_edge_transform_and_lower, "to_edge_transform_and_lower not found"
assert EdgeCompileConfig,           "EdgeCompileConfig not found"

# ── Backend partitioners (resolve only what we need) ─────────────────────
if RECIPE == "xnnpack":
    XnnpackPartitioner = _find("XnnpackPartitioner", [
        "executorch.backends.xnnpack.partition.xnnpack_partitioner",
    ])
    assert XnnpackPartitioner, "XnnpackPartitioner not found"

elif RECIPE == "cuda":
    CudaPartitioner = _find("CudaPartitioner", [
        "executorch.backends.cuda.cuda_partitioner",
        "executorch.backends.cuda.partition.cuda_partitioner",
    ])
    CompileSpec = _find("CompileSpec", [
        "executorch.exir.backend.compile_spec_schema",
    ])
    assert CudaPartitioner, (
        "CudaPartitioner not found. "
        "Ensure executorch is built with CUDA backend support:\n"
        "  pip install executorch[cuda]  OR  build from source with USE_CUDA=1"
    )

elif RECIPE == "vulkan":
    VulkanPartitioner = _find("VulkanPartitioner", [
        "executorch.backends.vulkan.partitioner.vulkan_partitioner",
        "executorch.backends.vulkan.vulkan_partitioner",
    ])
    assert VulkanPartitioner, (
        "VulkanPartitioner not found. "
        "Ensure executorch is built with Vulkan backend support:\n"
        "  pip install executorch[vulkan]  OR  build from source with USE_VULKAN=1"
    )

# ── Decide quantization path ────────────────────────────────────────────
USE_PT2E = (
    RECIPE == "xnnpack"
    and all([prepare_pt2e, convert_pt2e, XNNPACKQuantizer, get_sym_config])
)

path_label = (
    "PT2E + XNNPACKQuantizer"       if USE_PT2E else
    "Native INT8 module replacement"
)
print(f"\n  Quantization path : {path_label}")
print(f"  Backend recipe    : {RECIPE}")
print("Import resolution complete.")

# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODEL + TOKENIZER
# ═════════════════════════════════════════════════════════════════════════════

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is empty.\n"
        "  Set the HF_TOKEN environment variable before running:\n"
        "    export HF_TOKEN='hf_...'\n"
    )

login(token=HF_TOKEN, add_to_git_credential=False)
print("Logged in to HuggingFace.")

EXPORT_DTYPE = torch.float16   # fp16 for scales, norms, biases

is_local = os.path.isdir(MODEL_PATH)
print(f"Model : {MODEL_PATH}  ({'local' if is_local else 'HuggingFace Hub'})")
print(f"Dtype : {EXPORT_DTYPE}")

# ── Tokenizer ────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, token=HF_TOKEN if not is_local else None,
)

# ── Model ────────────────────────────────────────────────────────────────
print("\nLoading model weights...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    token=HF_TOKEN if not is_local else None,
    device_map="cpu",
    torch_dtype=EXPORT_DTYPE,
    use_cache=False,
    attn_implementation="eager",
)
base_model.eval()

n_params = sum(p.numel() for p in base_model.parameters())
print(f"  Parameters   : {n_params / 1e6:.1f} M")
print(f"  Linear layers: "
      f"{sum(1 for m in base_model.modules() if isinstance(m, torch.nn.Linear))}")

# ═════════════════════════════════════════════════════════════════════════════
# NATIVE INT8 MODULE REPLACEMENT
# ═════════════════════════════════════════════════════════════════════════════

class Int8Linear(torch.nn.Module):
    """nn.Linear replacement storing weights as int8."""
    def __init__(self, weight_int8, scale, bias):
        super().__init__()
        self.register_buffer("weight_q",     weight_int8)   # [out, in], int8
        self.register_buffer("weight_scale", scale)          # [out, 1],  fp16
        self.register_buffer("linear_bias",  bias)           # [out] or None

    def forward(self, x):
        w = self.weight_q.to(self.weight_scale.dtype) * self.weight_scale
        return F.linear(x, w, self.linear_bias)


class Int8Embedding(torch.nn.Module):
    """nn.Embedding replacement storing weights as int8."""
    def __init__(self, weight_int8, scale, padding_idx):
        super().__init__()
        self.register_buffer("weight_q",     weight_int8)   # [vocab, dim], int8
        self.register_buffer("weight_scale", scale)          # [vocab, 1],   fp16
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
            replacements.append(
                (name, Int8Linear(wq, sc.to(dtype), bias), "linear"))
        elif isinstance(module, torch.nn.Embedding):
            print(f"  Skipping {name} (keeping FP16)")
            continue

    for name, new_mod, _ in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    # Re-tie shared weights (e.g. lm_head <-> embed_tokens)
    embeds  = [(n, m) for n, m, t in replacements if t == "embed"]
    linears = [(n, m) for n, m, t in replacements if t == "linear"]
    tied = 0
    for en, em in embeds:
        for ln, lm in linears:
            if (lm.weight_q.shape == em.weight_q.shape
                    and torch.equal(lm.weight_q, em.weight_q)):
                lm.weight_q = em.weight_q
                lm.weight_scale = em.weight_scale
                tied += 1
                print(f"  Tied: {ln} <-> {en}")

    n_lin = sum(1 for _, _, t in replacements if t == "linear")
    n_emb = sum(1 for _, _, t in replacements if t == "embed")
    return n_lin, n_emb, tied


# ── Apply INT8 replacement (used for all recipes when PT2E is unavailable) ─
if not USE_PT2E:
    n_lin, n_emb, n_tied = replace_with_int8(base_model, EXPORT_DTYPE)

    # Size estimate
    int8_bytes, scale_bytes, other_bytes, seen = 0, 0, 0, set()
    for _, mod in base_model.named_modules():
        if isinstance(mod, (Int8Linear, Int8Embedding)):
            ptr = mod.weight_q.data_ptr()
            if ptr not in seen:
                int8_bytes  += mod.weight_q.numel()
                scale_bytes += (mod.weight_scale.numel()
                                * mod.weight_scale.element_size())
                seen.add(ptr)
            if isinstance(mod, Int8Linear) and mod.linear_bias is not None:
                other_bytes += (mod.linear_bias.numel()
                                * mod.linear_bias.element_size())
    for p in base_model.parameters():
        other_bytes += p.numel() * p.element_size()

    total = int8_bytes + scale_bytes + other_bytes
    print(f"\n  Replaced {n_lin} Linear + {n_emb} Embedding -> INT8")
    print(f"  Weight ties preserved: {n_tied}")
    print(f"  Estimated .pte size : ~{total / 1e9:.2f} GB")

# ── Export wrapper ───────────────────────────────────────────────────────
class GemmaExportWrapper(torch.nn.Module):
    """Tensor-in / tensor-out wrapper for ExecuTorch export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

wrapped_model = GemmaExportWrapper(base_model)
print("\nModel loaded, quantized (INT8), and wrapped.")

# ═════════════════════════════════════════════════════════════════════════════
# TRACE + LOWER + EXPORT .PTE
# ═════════════════════════════════════════════════════════════════════════════

print(f"Trace -> Lower ({RECIPE}) -> Save .pte\n")

# ── Example inputs + dynamic shapes ──────────────────────────────────────
dummy_ids  = torch.randint(0, tokenizer.vocab_size, (1, 128), dtype=torch.long)
dummy_mask = torch.ones(1, 128, dtype=torch.long)
example_args = (dummy_ids, dummy_mask)

seq = Dim("seq_len", min=1, max=MAX_SEQ)
dyn_shapes = ({1: seq}, {1: seq})

# ── Trace ────────────────────────────────────────────────────────────────
print("  Tracing model graph (high RAM -- several minutes)...")
exported_model = export(wrapped_model, example_args, dynamic_shapes=dyn_shapes)
print("  Traced.")

# ── PT2E quantization (only for xnnpack when XNNPACKQuantizer is found) ──
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

# ═════════════════════════════════════════════════════════════════════════════
# BACKEND-SPECIFIC LOWERING
# ═════════════════════════════════════════════════════════════════════════════

def lower_xnnpack(exported):
    """Lower to Edge IR using XNNPACK CPU backend."""
    print("  Lowering to Edge IR + XNNPACK backend...")
    return to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()


def lower_cuda(exported):
    """
    Lower to Edge IR using CUDA / AOTInductor backend.

    The CudaPartitioner delegates subgraphs to AOTInductor, which compiles
    them into optimised CUDA kernels. Ops not supported by the backend
    fall back to portable (CPU) kernels at runtime.
    """
    print("  Lowering to Edge IR + CUDA backend...")

    # CompileSpec list -- empty uses AOTInductor defaults (the partitioner
    # itself decides which ops to claim). Add entries here to tune:
    #   CompileSpec("option_name", b"value")
    compile_specs = []

    return to_edge_transform_and_lower(
        exported,
        partitioner=[CudaPartitioner(compile_specs)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()


def lower_vulkan(exported):
    """
    Lower to Edge IR using Vulkan compute-shader backend.

    VulkanPartitioner maps supported ops to SPIR-V compute shaders.
    Unsupported ops fall back to portable (CPU) kernels at runtime.
    Typical compile_options control texture limits and memory layout;
    None uses reasonable defaults.
    """
    print("  Lowering to Edge IR + Vulkan backend...")
    return to_edge_transform_and_lower(
        exported,
        partitioner=[VulkanPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).to_executorch()


lowering_fn = {
    "xnnpack": lower_xnnpack,
    "cuda":    lower_cuda,
    "vulkan":  lower_vulkan,
}

executorch_program = lowering_fn[RECIPE](exported_model)

# ═════════════════════════════════════════════════════════════════════════════
# SAVE .PTE
# ═════════════════════════════════════════════════════════════════════════════

_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", MODEL_PATH.rstrip("/").split("/")[-1])
_dtype_tag = "int8"

if args.output:
    output_filename = args.output
else:
    output_filename = f"{_slug}_{_dtype_tag}_{RECIPE}.pte"

with open(output_filename, "wb") as f:
    f.write(executorch_program.buffer)

size_mb = os.path.getsize(output_filename) / (1024 * 1024)
print(f"\n{'='*60}")
print(f"  SUCCESS")
print(f"  Model   : {MODEL_PATH}")
print(f"  Recipe  : {RECIPE}")
print(f"  File    : {os.path.abspath(output_filename)}")
print(f"  Size    : {size_mb:.2f} MB  ({size_mb/1024:.2f} GB)")
print(f"  Storage : INT8 weights (1 byte) + {EXPORT_DTYPE} norms/scales")
print(f"{'='*60}")

print(f"\nDeployment:")
if RECIPE == "xnnpack":
    print(f"  Target: CPU (ARM / x86 via XNNPACK)")
elif RECIPE == "cuda":
    print(f"  Target: NVIDIA GPU (CUDA via AOTInductor)")
elif RECIPE == "vulkan":
    print(f"  Target: GPU (Vulkan -- Android / desktop)")

print(f"\n  1. Copy {output_filename} to your app's assets/ folder")
print(f"  2. Load:  Module module = Module.load(assetFilePath(\"{output_filename}\"));")
print(f"  3. Run :  module.forward(inputIds, attentionMask);")

if RECIPE in ("cuda", "vulkan"):
    print(f"\n  Python inference:")
    print(f"    python run_inference_mixed_precision.py \\")
    print(f"        --model {output_filename} --prompt \"Hello\"")
