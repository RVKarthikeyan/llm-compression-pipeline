# Llama-3.2-1B-Instruct Compression Pipeline

A two-stage model compression pipeline that reduces **Llama-3.2-1B-Instruct** from 16 layers to 14 layers using Layer Collapse (LaCo) pruning, then recovers performance using LoRA fine-tuning.

---

## What This Does

| Stage | Technique | Purpose |
|-------|-----------|---------|
| Phase 1 | LaCo Layer Collapse | Prune 2 layer pairs (16 → 14 layers), reduce model size |
| Phase 2 | LoRA Fine-tuning | Heal the performance lost from pruning |
| Phase 3 | LoRA Merge + Export | Bake LoRA weights into base model, save final model |

**End result:** A compressed Llama model with fewer layers, smaller file size, faster inference, and recovered quality — exported as `.safetensors` and optionally uploaded to Google Drive for sharing.

---

## Requirements

### 1. Hardware

| GPU | VRAM | Status |
|-----|------|--------|
| T4 | 16 GB | Supported (Colab Free/Pro) |
| L4 | 24 GB | Supported (Colab Pro+) |
| V100 | 16 GB | Supported (Colab Pro) |
| A100 | 40 GB | Supported (Colab Pro+) |
| H100 | 80 GB | Supported (Colab Pro+) |

> No GPU = pipeline runs but with minimal settings (100 samples, batch size 1). Not recommended for real use.

### 2. Software (auto-installed in notebook)

```
transformers
accelerate
peft
datasets
bitsandbytes
sentencepiece
protobuf
huggingface_hub
```

### 3. Accounts Needed

- **Hugging Face account** — to download the model
- **Google account** — for Google Colab and Google Drive
- **Meta license approval** — required to access Llama models (free, auto-approved)

---

## Setup Before Running

### Step 1: Accept Meta License

1. Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Log in to your Hugging Face account
3. Fill in the license form (name, country, intended use)
4. Click Submit — approval is usually instant
5. You will see: "You have been granted access to this model"

### Step 2: Get a Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: anything (e.g. `colab-llama`)
4. Type: `Read`
5. Click "Generate a token"
6. Copy the token (starts with `hf_...`) — you only see it once

### Step 3: Add Token to Colab Secrets

1. Open the notebook in Google Colab
2. Click the key icon in the left sidebar (Secrets)
3. Click "Add new secret"
4. Name: `HF_TOKEN`
5. Value: paste your `hf_...` token
6. Toggle "Notebook access" ON

---

## How to Run

1. Upload `colab_notebook_with_auth.ipynb` to Google Colab
   - File → Upload notebook → select the file
2. Set runtime to GPU
   - Runtime → Change runtime type → GPU → Save
3. Run all cells top to bottom
   - Runtime → Run all

The notebook will handle everything automatically.

---

## Pipeline Walkthrough

### Step 0: Authentication
Loads your HF token from Colab Secrets and logs in to Hugging Face.

### Step 1: Install Libraries
Installs all required Python packages silently. Detects and prints GPU info.

### Step 2: Configuration
Auto-detects your GPU and sets optimal batch size, sequence length, and dataset size.

```
Base model    : meta-llama/Llama-3.2-1B-Instruct
Layers merged : (2,3) and (7,8)  →  16 layers become 14
Dataset       : timdettmers/openassistant-guanaco
LoRA rank     : 16
LoRA alpha    : 32
Learning rate : 2e-4
```

### Step 3: Layer Collapse Class
Defines `LayerCollapser` which:
- Finds adjacent layer pairs
- Merges them by averaging all weights
- Reassigns `layer_idx` on attention modules (critical for KV cache correctness)

### Step 4: Helper Functions
- `load_and_prepare_dataset()` — loads and tokenizes the Guanaco dataset
- `test_model()` — runs inference using the Llama-3 chat template format

### Phase 1: LaCo Pruning
1. Loads `meta-llama/Llama-3.2-1B-Instruct` (~2.5 GB download)
2. Tests the model before pruning
3. Merges layer pairs (2,3) and (7,8) by weight averaging
4. Reassigns layer indices to fix KV cache
5. Updates `config.num_hidden_layers` from 16 to 14
6. Tests after pruning (some quality degradation is expected)
7. Saves pruned model to `./compressed_llama/`

### Phase 2: LoRA Fine-tuning
1. Loads the Guanaco instruction dataset
2. Applies LoRA adapters to all projection layers:
   `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
3. Fine-tunes for 3 epochs using AdamW + cosine LR schedule
4. Saves LoRA adapter weights to `./lora_adapters/`
5. Tests healed model (quality should be close to original)

### Phase 3: Merge and Export
1. Reloads the pruned base model from disk
2. Loads the LoRA adapters
3. Merges LoRA into base model with `merge_and_unload()`
4. Saves final model to `./final_healed_model/` in safetensors format
5. Creates `compressed_llama_final.zip` archive
6. Runs two final inference tests

### Step 5: Summary
Prints sizes of pruned model and final healed model in GB.

### Step 5b: Upload to Google Drive
1. Mounts your Google Drive
2. Creates folder `MyDrive/Llama-3.2-1B-Compressed/`
3. Copies zip archive and individual model files to Drive
4. Prints instructions for sharing with others

### Step 6: Download
Downloads the zip file directly to your browser's download folder.

---

## Output Files

```
compressed_llama/          # Pruned model (16 → 14 layers)
    config.json
    model.safetensors
    tokenizer files...

lora_adapters/             # LoRA adapter weights only
    adapter_config.json
    adapter_model.safetensors

final_healed_model/        # Final merged model (ready to use)
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    generation_config.json

compressed_llama_final.zip # Zip of final_healed_model for sharing
```

---

## Using the Final Model

### Load from local folder

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./final_healed_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./final_healed_model")
```

### Load from Google Drive (in Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/content/drive/MyDrive/Llama-3.2-1B-Compressed/final_healed_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/content/drive/MyDrive/Llama-3.2-1B-Compressed/final_healed_model"
)
```

### Run inference

```python
messages = [{"role": "user", "content": "What is machine learning?"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

---

## Sharing the Model with Others

After Step 5b completes:

1. Go to https://drive.google.com
2. Navigate to `MyDrive/Llama-3.2-1B-Compressed`
3. Right-click the folder → Share → "Anyone with the link"
4. Copy and share the link

Recipients can either:
- Download the zip file and use the model locally
- Mount the shared Drive in their own Colab and load directly

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `GatedRepoError: 403 Forbidden` | Meta license not accepted | Visit the model page and accept the license, then re-run |
| `IndexError: list index out of range` | KV cache layer index mismatch after collapse | Fixed in current version — layer_idx is reassigned after collapse |
| `HF_TOKEN secret is empty` | Token not added to Colab Secrets | Add HF_TOKEN in the Colab key icon sidebar |
| `CUDA out of memory` | Batch size too large for GPU | Reduce `per_device_train_batch_size` in CONFIG |
| Phase 3 load fails | `num_hidden_layers` mismatch in config | Fixed in current version — config is updated after collapse |

---

## Project Structure

```
gemma_compression_pipeline/
    colab_notebook_with_auth.ipynb   # Main notebook (run this)
    README.md                        # This file
    PRUNING_GUIDE.md                 # Theory and detailed explanation of LaCo + LoRA
```

---

## Key Concepts

**LaCo (Layer Collapse):** Reduces model depth by merging adjacent transformer layers. Merged weights are the element-wise average of both layers. This creates "weight shock" — temporary quality degradation that is fixed in Phase 2.

**LoRA (Low-Rank Adaptation):** Adds small trainable matrices alongside frozen base weights. Only ~1-3% of parameters are trained, making fine-tuning fast and memory-efficient. After training, LoRA weights are merged back into the base model.

**safetensors:** The output format for all saved models. Faster and safer than PyTorch's `.bin` pickle format — no arbitrary code execution risk during loading.
