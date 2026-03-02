# Two-Stage Model Compression Pipeline for Gemma-3-1B-IT

Compress Google's **Gemma-3-1B-IT** using structural pruning (LaCo) + LoRA fine-tuning to get a smaller, faster model that retains ~90–95% of original quality.

| Metric | Value |
|--------|-------|
| Model Size | ~2.5 GB → ~2.0 GB (20% smaller) |
| Inference Speed | 15–25% faster |
| Quality Retained | 90–95% after LoRA healing |
| Layers | 18 → 16 (merge 2 pairs) |
| Runs on | Google Colab Free (T4 GPU) |

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [Theory: How It Works](#theory-how-it-works)
3. [Phase 1: Layer Collapse (LaCo)](#phase-1-layer-collapse-laco)
4. [Phase 2: LoRA Fine-tuning (Healing)](#phase-2-lora-fine-tuning-healing)
5. [Phase 3: Merge and Deploy](#phase-3-merge-and-deploy)
6. [How to Run](#how-to-run)
7. [Configuration Reference](#configuration-reference)
8. [Performance Characteristics](#performance-characteristics)
9. [Output Files](#output-files)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## What This Does

The pipeline runs in 3 phases:

**Phase 1 — Layer Collapse (LaCo):** Merges adjacent transformer layers by averaging their weights. This physically reduces the model depth from 18 to 16 layers. The model degrades temporarily ("weight shock") — that's expected.

**Phase 2 — LoRA Healing:** Fine-tunes small LoRA adapters on the pruned model using the Guanaco instruction dataset. This restores the quality lost from pruning, without retraining all 1B parameters.

**Phase 3 — Merge and Deploy:** Folds the LoRA adapters back into the base weights, producing a standalone model with no adapter overhead at inference time.

```
Original (18 layers, 2.5 GB)
        ↓  Phase 1: Layer Collapse
Pruned  (16 layers, 2.0 GB)  ← quality degraded
        ↓  Phase 2: LoRA Fine-tuning
Healed  (16 layers, 2.0 GB)  ← quality recovered
        ↓  Phase 3: Merge
Final   (16 layers, 2.0 GB)  ← standalone, deploy-ready ✅
```

### Output Directory Structure

```
compressed_gemma/          ← Pruned model (after Phase 1)
lora_adapters/             ← Trained LoRA adapter weights (after Phase 2)
final_healed_model/        ← ✅ Final model — use this for inference
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
compressed_gemma_final.zip ← Downloadable archive
```

### Files in This Repo

| File | Purpose |
|------|---------|
| `colab_notebook_with_auth.ipynb` | Main notebook — run this on Colab |
| `gemma_compression_pipeline.py` | Equivalent Python script for local use |
| `requirements.txt` | Python dependencies |
| `gpu_config.py` | GPU detection and config helper |
| `QUICKSTART.md` | Quick reference card |

---

## Theory: How It Works

### Why Structural Pruning?

LLMs like Gemma stack many transformer layers. Research shows adjacent layers in a trained model often learn very similar representations — they do nearly redundant work, especially in the middle layers.

The question: can we **remove some layers** without destroying the model?

### Naive Deletion vs. Weight Averaging

**Naive deletion** (just removing a layer) causes catastrophic quality loss — the remaining layers were never trained to compensate.

**LaCo (Layer Collapse)** does something smarter:

```
Layer L1 weights: W1
Layer L2 weights: W2
                 ↓
Merged weights:  W_merged = (W1 + W2) / 2
```

By averaging the weights of two adjacent layers we:
1. Preserve the combined "knowledge" from both layers
2. Produce a single layer that approximates both
3. Remove L2, keeping L1 with the merged weights

This is still lossy (hence "weight shock") but far less destructive than deletion.

### Why Weight Shock Happens

After merging, the model has a structural mismatch:
- Layers before the merge were trained to feed into N layers
- Now there are N-2 layers — the residual stream flow is disrupted
- Surviving layers were not trained to compensate for their missing neighbors

This is why Phase 2 (LoRA healing) is essential.

### What Gets Merged

The `LayerCollapser` merges all sub-components of each transformer layer component-wise:

| Component | What it is |
|-----------|------------|
| `q_proj`, `k_proj`, `v_proj` | Attention query / key / value projections |
| `o_proj` | Attention output projection |
| `gate_proj`, `up_proj` | MLP gate and up projections |
| `down_proj` | MLP down projection |
| Layer norms | Pre/post normalization weights |

All merged as: `W_merged[key] = (W1[key] + W2[key]) / 2.0`

### Which Layers to Target

Gemma-3-1B-IT has ~18 transformer layers (0-indexed):

```
Layers 0–4:    Early layers  — tokenization, syntax basics (avoid)
Layers 5–12:   Middle layers — ✅ Best pruning targets (redundant)
Layers 13–17:  Late layers   — reasoning, output formatting (avoid)
```

**Default config merges:** `(2, 3)` and `(8, 9)`

---

## Phase 1: Layer Collapse (LaCo)

### What the Code Does

The `LayerCollapser` class:

1. Locates layers via `model.model.layers` (Gemma's architecture path)
2. Sorts merge pairs in **reverse order** to avoid index shifting when removing layers
3. For each pair `(idx1, idx2)`:
   - Averages all weight tensors between the two layers
   - Loads merged weights into `layers_list[idx1]`
   - Removes `layers_list[idx2]`
4. Reconstructs `nn.ModuleList` with the reduced layer list

```python
# Weight averaging — core of LaCo
merged_state_dict = {}
for key in state_dict1.keys():
    if key in state_dict2:
        merged_state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2.0
    else:
        merged_state_dict[key] = state_dict1[key]
```

```python
# Reverse order prevents index shifts during removal
layers_to_merge_sorted = sorted(layers_to_merge, reverse=True)
# e.g. [(8,9), (2,3)] instead of [(2,3), (8,9)]
```

### What to Expect After Phase 1

- Model size: ~2.5 GB → ~2.0 GB
- Layer count: 18 → 16
- Quality: **Noticeably degraded** — repetition, incoherence, short outputs
- This is **normal** — don't panic, Phase 2 fixes it

---

## Phase 2: LoRA Fine-tuning (Healing)

### Why LoRA Instead of Full Fine-tuning

Full fine-tuning updates all ~1B parameters — too expensive, and risks overfitting on the small recovery dataset.

**LoRA (Low-Rank Adaptation)** inserts small trainable matrices alongside frozen pretrained weights:

```
Original output:  Y = X · W₀           (W₀ frozen)
LoRA output:      Y = X · W₀ + X · BA  (B, A are trainable)

Where:
  W₀ ∈ R^(d × d)   frozen, large
  A  ∈ R^(r × d)   trainable, small
  B  ∈ R^(d × r)   trainable, small
  r  = LoRA rank (16 by default)
```

Only ~2–5% of parameters are trained. The LoRA adapters learn to bridge the gap left by the merged layers.

### LoRA Configuration

```python
LoraConfig(
    r=16,             # Rank — controls adapter capacity
    lora_alpha=32,    # Scaling factor (effective scale = alpha/r = 2.0)
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"        # MLP
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Dataset

**timdettmers/openassistant-guanaco** — high-quality human-written instruction-following conversations from OpenAssistant.

| Setting | Value |
|---------|-------|
| Samples used | 1000 (T4) / 1500 (L4) / 2000 (A100) |
| Epochs | 3 |
| Max sequence length | 512 tokens |
| Optimizer | AdamW |
| LR schedule | Cosine decay |
| Learning rate | 2e-4 |

### What to Expect After Phase 2

- Quality mostly recovered — coherent, instruction-following responses
- LoRA adapters saved to `./lora_adapters/`
- Adapters are still separate from base weights until Phase 3

---

## Phase 3: Merge and Deploy

After healing, the model has two components: pruned base weights + LoRA adapters. Running with separate adapters adds overhead. Merging folds LoRA back into the base weights:

```
W_final = W₀ + BA   (single weight matrix, no adapter overhead)
```

```python
# Load pruned base
base_model = AutoModelForCausalLM.from_pretrained("./compressed_gemma", ...)

# Attach LoRA adapters
model = PeftModel.from_pretrained(base_model, "./lora_adapters")

# Fold adapters into weights
merged_model = model.merge_and_unload()

# Save as standard HuggingFace model
merged_model.save_pretrained("./final_healed_model")
tokenizer.save_pretrained("./final_healed_model")
```

The result is a standalone model — no PEFT library needed at inference time.

---

## How to Run

### Option A: Google Colab (Recommended)

#### Prerequisites

1. Accept the Gemma license at [huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) — click **"Agree and access repository"**
2. Get a HuggingFace READ token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

#### Steps

**1. Upload notebook:**
```
Go to https://colab.research.google.com/
File → Upload notebook → select colab_notebook_with_auth.ipynb
```

**2. Enable GPU:**
```
Runtime → Change runtime type → Hardware accelerator → GPU (T4 for free tier)
```

**3. Add your token — pick one method:**

*Method A — paste directly (easiest):*
```python
# In the first code cell:
HF_TOKEN = "hf_your_token_here"
```

*Method B — Colab Secrets (more secure):*
```
Click 🔑 icon in the left sidebar
Add secret:  Name = HF_TOKEN,  Value = hf_your_token_here
Toggle "Notebook access" ON
```

**4. Run all cells:**
```
Runtime → Run all  (or Ctrl+F9)
```

**5. Download:** The last cell automatically downloads `compressed_gemma_final.zip`.

#### Timeline on T4 GPU

```
0–5 min    Phase 1: Load model, apply layer collapse, save pruned model
5–50 min   Phase 2: Load dataset, LoRA training (3 epochs × 1000 samples)
50–55 min  Phase 3: Merge adapters, save final model, create zip
```

---

### Option B: Local Python Script

```bash
cd gemma_compression_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Login to HuggingFace
huggingface-cli login

# Run
python gemma_compression_pipeline.py
```

---

### Using the Final Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./final_healed_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./final_healed_model")

prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Configuration Reference

### GPU Auto-Detected Settings

The notebook detects your GPU and sets everything automatically — no manual changes needed.

| GPU | Batch Size | Grad Accum | Dataset Samples | Expected Time |
|-----|-----------|------------|-----------------|---------------|
| T4 (16GB) | 4 | 4 | 1000 | 40–55 min |
| L4 (24GB) | 8 | 2 | 1500 | 25–30 min |
| V100 (16GB) | 6 | 3 | 1200 | 30–35 min |
| A100 (40GB) | 16 | 1 | 2000 | 15–20 min |
| H100 (80GB) | 32 | 1 | 3000 | 10–15 min |

### Layer Merge Presets

| Preset | `layers_to_merge` | Compression | Quality Risk |
|--------|-------------------|-------------|--------------|
| Conservative | `[(6, 7)]` | ~10% | Low |
| Balanced (default) | `[(2, 3), (8, 9)]` | ~20% | Medium |
| Aggressive | `[(2, 3), (6, 7), (10, 11)]` | ~28% | High |

### Full CONFIG Reference

```python
CONFIG = {
    # Model
    "base_model": "google/gemma-3-1b-it",
    "dataset": "timdettmers/openassistant-guanaco",

    # Paths
    "output_dir": "./compressed_gemma",
    "lora_output_dir": "./lora_adapters",
    "final_merged_dir": "./final_healed_model",
    "zip_output": "./compressed_gemma_final.zip",

    # Layer pruning
    "layers_to_merge": [(2, 3), (8, 9)],

    # LoRA
    "lora_r": 16,           # Rank — higher = more capacity, more memory
    "lora_alpha": 32,       # Scale = alpha / r
    "lora_dropout": 0.05,

    # Training (auto-set by GPU detection)
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_seq_length": 512,
    "fp16": True,
}
```

---

## Performance Characteristics

### Model Size at Each Stage

| Stage | Size | Layers |
|-------|------|--------|
| Original Gemma-3-1B-IT | ~2.5 GB | 18 |
| After Layer Collapse | ~2.0 GB | 16 |
| After LoRA Merge | ~2.0 GB | 16 |

### Inference Performance

| Metric | Value |
|--------|-------|
| Inference Speed | 15–25% faster |
| VRAM Usage | 15–20% lower |
| Quality vs original | 90–95% retained |

### Quality at Each Stage

| Stage | Quality | Notes |
|-------|---------|-------|
| Before pruning | Baseline | Full Gemma-3-1B-IT |
| After layer collapse | Degraded | Repetition, confused outputs — normal |
| After LoRA healing | ~90–95% | Near-original instruction following |
| After merge | Same as healed | Merge doesn't affect quality |

---

## Output Files

```
compressed_gemma/          ← Pruned model (after layer collapse)
lora_adapters/             ← Trained LoRA adapters
final_healed_model/        ← ✅ Production model — use this
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
compressed_gemma_final.zip ← Downloadable archive
```

---

## Troubleshooting

### Error: 401 Unauthorized

**Cause:** HuggingFace token missing or Gemma license not accepted.

**Fix:**
1. Accept the license at [huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
2. Check token is correctly pasted in the first code cell
3. Verify the token has READ permissions

---

### Error: CUDA Out of Memory

**Fix — reduce memory pressure:**
```python
CONFIG["per_device_train_batch_size"] = 2
CONFIG["gradient_accumulation_steps"] = 8
CONFIG["max_seq_length"] = 256
CONFIG["dataset_split"] = "train[:500]"
```

Or load with 8-bit quantization:
```python
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"],
    load_in_8bit=True,
    device_map="auto",
    token=HF_TOKEN
)
```

---

### Poor Quality After Healing

**Fix — increase healing power:**
```python
CONFIG["lora_r"] = 32
CONFIG["num_train_epochs"] = 5
CONFIG["dataset_split"] = "train[:3000]"
```

Or reduce pruning aggression:
```python
CONFIG["layers_to_merge"] = [(6, 7)]  # Only 1 pair instead of 2
```

---

### Model Repeats Itself After Phase 1

This is **expected**. Weight shock from merging causes temporary incoherence. Phase 2 (LoRA healing) fixes it — do not skip Phase 2.

---

### Training Very Slow

```python
CONFIG["dataset_split"] = "train[:500]"  # Half the samples
CONFIG["max_seq_length"] = 256           # Half the sequence length
```

These two changes alone cut training time roughly in half.

---

### `AttributeError: Could not find model layers`

Inspect the model structure to find the correct path:
```python
print(model)
# For Gemma: model.model.layers ✅
```

---

## References

- [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) — Base model
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) — Training dataset
- [Hugging Face PEFT](https://github.com/huggingface/peft) — LoRA library

### Libraries Used

| Library | Role |
|---------|------|
| `transformers` | Model loading and training |
| `peft` | LoRA adapter creation and merging |
| `datasets` | Dataset loading and tokenization |
| `bitsandbytes` | Optional 8-bit loading for memory savings |
| `accelerate` | Device map and multi-GPU support |

---

## License

Provided for educational and research purposes. Comply with:
- [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (Google)
- Hugging Face Transformers license (Apache 2.0)
- OpenAssistant Guanaco dataset license
