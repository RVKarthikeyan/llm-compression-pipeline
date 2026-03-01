# Custom Mixed-Precision Quantization for ExecuTorch — Theory & Guide

## Table of Contents

1. [Overview](#overview)
2. [Why Quantization?](#why-quantization)
3. [Number Formats: FP32 → FP16 → INT8](#number-formats-fp32--fp16--int8)
4. [Per-Channel Symmetric Quantization](#per-channel-symmetric-quantization)
5. [Mixed Precision — What Lives Where](#mixed-precision--what-lives-where)
6. [Weight Tying Optimization](#weight-tying-optimization)
7. [The ExecuTorch Pipeline](#the-executorch-pipeline)
8. [Why Not Use torchao / PT2E?](#why-not-use-torchao--pt2e)
9. [Size Breakdown (Gemma 3 1B)](#size-breakdown-gemma-3-1b)
10. [How to Run the Notebook](#how-to-run-the-notebook)
11. [Android Deployment](#android-deployment)
12. [References](#references)

---

## Overview

This notebook converts a **Gemma-family** language model into a **.pte file** that
runs on Android phones via Meta's **ExecuTorch** runtime.  The core idea is
simple:

> Store every weight as an **8-bit integer** (1 byte) instead of a 16-bit or
> 32-bit float (2–4 bytes), then dequantise back to float only when computing
> a layer's forward pass.

The result for a 1-billion-parameter model:

| Format | Bytes / param | .pte size |
|--------|---------------|-----------|
| FP32   | 4             | ~5.0 GB   |
| FP16   | 2             | ~2.5 GB   |
| **INT8** | **1**        | **~0.93 GB** |

---

## Why Quantization?

Large language models are trained in FP32 or BF16 — formats that give the
optimizer fine-grained gradients. But at **inference** time:

1. **Memory** — A phone has 4–12 GB of RAM shared with the OS, camera, and
   apps.  A 5 GB model simply won't fit alongside everything else.
2. **Bandwidth** — Moving data from RAM to the CPU/GPU is the bottleneck for
   transformer inference (it's *memory-bound*).  Smaller weights → more tokens
   per second.
3. **Download** — Users don't want to download 5 GB over cellular data.

Quantization trades a tiny amount of numerical precision for a dramatic
reduction in size and latency.

---

## Number Formats: FP32 → FP16 → INT8

### FP32 (float32) — 4 bytes

```
[1 sign][8 exponent][23 mantissa]  →  ±3.4 × 10³⁸, ~7 decimal digits
```

Full training precision. Every operation is exact to ~7 significant figures.

### FP16 (float16) — 2 bytes

```
[1 sign][5 exponent][10 mantissa]  →  ±65504, ~3.3 decimal digits
```

Half the memory, and modern CPUs (ARM NEON) have native FP16 instructions.
Most post-training weights live comfortably in this range.

### INT8 (int8) — 1 byte

```
[8 bits, two's complement]  →  {-128, -127, …, 0, …, 126, 127}
```

Only 256 possible values. To represent a continuous weight distribution we need
a **scale factor** that maps this discrete grid back to floating point. That's
quantization.

---

## Per-Channel Symmetric Quantization

We use **symmetric, per-channel (per-row)** quantization — the most common
scheme for weight-only quantization.

### The Math

For a weight matrix **W** of shape `[out_features, in_features]`:

1. **Compute the scale** for each output channel (row):

$$
s_i = \frac{\max(|W_{i,:}|)}{127}
$$

2. **Quantize** (float → int8):

$$
W^q_{i,j} = \text{round}\!\left(\frac{W_{i,j}}{s_i}\right), \quad \text{clamped to } [-128,\; 127]
$$

3. **Dequantize** (int8 → float, at inference time):

$$
\hat{W}_{i,j} = W^q_{i,j} \times s_i
$$

### Why "Symmetric"?

The zero point is always 0 — we don't store an offset. This simplifies the
dequantize kernel and halves the metadata. For normally-distributed weights
(centred near zero) this is a good fit.

### Why "Per-Channel"?

Different rows of a weight matrix can have very different magnitudes.  A single
global scale would waste precision on small-magnitude rows.  Per-channel gives
each row its own scale, preserving more information.

### Storage Cost

- **Weights:** `out × in` values × 1 byte = `out × in` bytes
- **Scales:** `out` values × 2 bytes (FP16) = `2 × out` bytes
- **Overhead:** The scale vector is tiny — e.g., for a 2048×2048 matrix the
  scales add only 4 KB to the 4 MB of int8 weights (0.1%).

---

## Mixed Precision — What Lives Where

Not everything gets quantized. The notebook applies **mixed precision**:

| Component | Stored as | Why |
|-----------|-----------|-----|
| `nn.Linear` weights | **int8** + fp16 scale | Bulk of parameters; tolerant to quantization |
| `nn.Embedding` weights | **int8** + fp16 scale | Large vocab table; same technique works |
| `Linear` biases | **fp16** | Tiny vectors — quantizing saves almost nothing |
| `LayerNorm` / `RMSNorm` weights | **fp16** | Normalisation is sensitive to precision |
| Attention scales, rotary embeddings | **fp16** | Structural constants — must stay accurate |

This is "mixed" because quantized int8 coexists with full-precision fp16 in the
same model.  The dequantize operation `int8 × scale → fp16` happens inside
each module's `forward()`, so from the rest of the graph's perspective every
tensor is fp16.

---

## Weight Tying Optimization

Many language models **tie** the input embedding and the output projection:

```
model.embed_tokens.weight  is  model.lm_head.weight   # same tensor in memory
```

After quantization, both become `Int8Embedding` and `Int8Linear` with
*separate* int8 buffers. If we don't reconnect them, `torch.export` serializes
the same data twice — doubling the embedding's contribution to the .pte file.

The notebook detects tied weights by checking shape and value equality, then
reassigns the buffer pointers:

```python
lm_head.weight_q     = embed_tokens.weight_q      # same underlying storage
lm_head.weight_scale = embed_tokens.weight_scale
```

`torch.export` sees one buffer with two references and serializes it once.

---

## The ExecuTorch Pipeline

The journey from PyTorch model to on-device binary has four stages:

```
 PyTorch nn.Module
       │
       ▼
 ① torch.export.export()       → ExportedProgram (ATen IR)
       │
       ▼
 ② to_edge_transform_and_lower → Edge IR + XNNPACK-delegated subgraphs
       │
       ▼
 ③ .to_executorch()            → ExecutorchProgram (flatbuffer)
       │
       ▼
 ④ .buffer  →  write to .pte file
```

### Stage 1 — `torch.export`

Symbolically traces the model, producing a graph of ATen operators (matmul,
add, etc.) with no Python control flow. Dynamic shapes (variable sequence
length) are captured via `torch.export.Dim`.

### Stage 2 — Edge + XNNPACK Lowering

`to_edge_transform_and_lower` does two things:

- **Edge transform** — converts ATen ops to a smaller, mobile-friendly op set.
- **XNNPACK partitioner** — identifies subgraphs that the XNNPACK library can
  accelerate (optimised ARM NEON/SSE kernels for conv, matmul, etc.) and
  delegates them. Non-delegated ops run on a portable CPU interpreter.

### Stage 3 — Serialization

`.to_executorch()` serializes the graph, tensor data, and delegation metadata
into a FlatBuffer — the `.pte` format. Buffer pointers become file offsets.

### Stage 4 — On-Device Loading

On Android, the ExecuTorch runtime memory-maps the `.pte` file, sets up the
XNNPACK delegate, and exposes a `forward(input_ids, attention_mask) → logits`
interface.

---

## Why Not Use torchao / PT2E?

The "official" quantization paths in the PyTorch ecosystem are:

1. **torchao `int8_weight_only()`** — calls C++ kernels to pack int8. But
   torchao's C++ extensions must be ABI-compatible with the exact torch build.
   On Colab's nightly torch, the extensions fail to load, and the function
   silently becomes a no-op (weights stay FP32).

2. **PT2E (`prepare_pt2e` / `convert_pt2e`)** — inserts fake-quantize observers
   into the exported graph. Requires `torch.ao.quantization.pt2e` which is
   missing from current nightly builds.

Our approach sidesteps both by doing the quantization **ourselves**: replace
`nn.Linear` with a custom `Int8Linear` that stores `torch.int8` buffers
directly. No C++ extensions needed, no compatibility issues, and the int8
tensors flow straight into the .pte file at 1 byte each.

---

## Size Breakdown (Gemma 3 1B)

`google/gemma-3-1b-it` has ~1.0 billion parameters:

| Category | Parameters | Storage | Size |
|----------|-----------|---------|------|
| Linear weights (int8) | ~990 M | 1 byte each | ~943 MB |
| Embedding weights (int8) | ~524 M* | 1 byte each | ~500 MB* |
| *Tied with lm_head* | −524 M* | *shared buffer* | *−500 MB* |
| Scales (fp16) | ~0.5 M | 2 bytes each | ~1 MB |
| Biases + norms (fp16) | ~0.3 M | 2 bytes each | ~0.6 MB |
| Graph metadata | — | — | ~10 MB |
| **Total** | | | **~955 MB (0.93 GB)** |

*The embedding and lm_head share weights (weight tying), so the 524 M
embedding parameters are counted once.*

---

## How to Run the Notebook

### Prerequisites

- **Google Colab** with **TPU v5e-1** runtime (for ≥48 GB system RAM).
  - T4 GPU (~12 GB) will OOM during `torch.export`.
- A **HuggingFace account** with access to the target model.

### Step-by-Step

1. **Open** the notebook in Google Colab.
2. **Set runtime** → `TPU v5e-1` (or any runtime with ≥ 48 GB RAM).
3. **Edit Cell 5** → set `MODEL_PATH` to your model:
   ```python
   MODEL_PATH = "google/gemma-3-1b-it"            # HuggingFace
   MODEL_PATH = "/content/drive/MyDrive/my_model"  # Local/Drive
   ```
4. **Set HF token** (Cell 4) via one of:
   - **Colab Secrets** (recommended): Add a secret named `HF_TOKEN` in the 🔑
     sidebar.
   - **Direct paste**: Set `HF_TOKEN = "hf_your_token_here"` in Cell 4.
5. **Run Cell 2** (installation). Wait for it to complete (~3 minutes).
6. **Runtime → Restart session** (required to pick up new torch).
7. **Run Cells 3 → 4 → 5 → 6** in order.
8. Cell 6 will print the output filename and size. Download the `.pte` file.

### Expected Output

```
============================================================
  ✅  SUCCESS!
  Model  : google/gemma-3-1b-it
  File   : gemma-3-1b-it_int8.pte
  Size   : 956.19 MB  (0.93 GB)
  Storage: INT8 weights (1 byte) + torch.float16 norms/scales
============================================================
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| OOM / process killed | Switch to TPU v5e or higher RAM runtime |
| `ModuleNotFoundError: executorch` | Re-run Cell 2, then restart runtime |
| HF `401 Unauthorized` | Check token; accept model license on HF website |
| `FutureWarning: isinstance(treespec, LeafSpec)` | Harmless — ignore |

---

## Android Deployment

Once you have the `.pte` file:

1. **Add ExecuTorch AAR** to your Android project:
   ```gradle
   implementation 'org.pytorch:executorch-android:0.6.0'
   ```

2. **Copy the .pte** to `app/src/main/assets/`.

3. **Load and run** in Java/Kotlin:
   ```java
   import org.pytorch.executorch.Module;
   import org.pytorch.executorch.Tensor;

   Module module = Module.load(assetFilePath("gemma-3-1b-it_int8.pte"));

   long[] inputIds = tokenizer.encode(userPrompt);
   long[] mask     = new long[inputIds.length];
   Arrays.fill(mask, 1L);

   Tensor result = module.forward(
       Tensor.fromBlob(inputIds, new long[]{1, inputIds.length}),
       Tensor.fromBlob(mask,     new long[]{1, mask.length})
   );
   // result contains logits — apply argmax / sampling for next token
   ```

4. **Autoregressive loop**: Call `forward()` repeatedly, appending each
   generated token to the input sequence, until you hit the EOS token or
   your max length.

---

## References

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [ExecuTorch Android Tutorial](https://pytorch.org/executorch/stable/llm/llm-manual-android.html)
- [torch.export](https://pytorch.org/docs/stable/export.html)
- [XNNPACK Backend](https://pytorch.org/executorch/stable/backends-xnnpack.html)
- [Quantization Concepts (PyTorch)](https://pytorch.org/docs/stable/quantization.html)
- [Gemma Models (Google)](https://ai.google.dev/gemma)
- [torchao](https://github.com/pytorch/ao)
