"""
Two-Stage Model Compression Pipeline for Gemma-3-1B-IT
=======================================================

Phase 1: Architecture Optimization (Pruning via LaCo)
  - Layer Collapse: merge adjacent layers by averaging weights

Phase 2: Knowledge Recovery (Healing via LoRA)
  - Fine-tune with LoRA adapters on instruction-following dataset

Phase 3: Merge and Deploy
  - Merge LoRA adapters back into base model and save

Prerequisites
-------------
    pip install transformers accelerate peft datasets bitsandbytes sentencepiece protobuf huggingface_hub

Usage
-----
    Set HF_TOKEN env variable or edit the HF_TOKEN line below.
    python pruning_gemma3_1b_it.py

    Or import and call:
        from pruning_gemma3_1b_it import run_pruning_pipeline
        result = run_pruning_pipeline(hf_token="hf_...", model_path="google/gemma-3-1b-it")
"""

import os
import shutil
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel


# ─────────────────────────────────────────────
# 1. GPU Detection and Configuration
# ─────────────────────────────────────────────

def get_gpu_optimized_config(
    model_path: str = "google/gemma-3-1b-it",
    output_dir: str = "./compressed_gemma",
    lora_output_dir: str = "./lora_adapters",
    final_merged_dir: str = "./final_healed_model",
    zip_output: str = "./compressed_gemma_final.zip",
    layers_to_merge: Optional[List[Tuple[int, int]]] = None,
    dataset: str = "timdettmers/openassistant-guanaco",
    config_overrides: Optional[dict] = None,
) -> dict:
    """Auto-detect GPU and return optimized training configuration."""

    base_config = {
        "base_model": model_path,
        "dataset": dataset,
        "output_dir": output_dir,
        "lora_output_dir": lora_output_dir,
        "final_merged_dir": final_merged_dir,
        "zip_output": zip_output,

        # Layer pairs to merge (0-indexed). Gemma-3-1B-IT has 26 layers.
        "layers_to_merge": layers_to_merge if layers_to_merge is not None else [(2, 3), (8, 9)],

        # LoRA
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],

        # Training
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 100,
    }

    if not torch.cuda.is_available():
        print("⚠️  No GPU detected — using minimal CPU settings")
        base_config.update({
            "dataset_split": "train[:100]",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 256,
            "fp16": False,
        })
    else:
        gpu_name = torch.cuda.get_device_name(0).lower()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"GPU: {torch.cuda.get_device_name(0)}  ({gpu_memory:.1f} GB)")

        if "t4" in gpu_name:
            print("⚙️  Optimizing for T4 — estimated training time: ~40-55 min")
            base_config.update({
                "dataset_split": "train[:1000]",
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 256,
                "fp16": True,
            })
        elif "l4" in gpu_name:
            print("⚙️  Optimizing for L4 — estimated training time: ~25-30 min")
            base_config.update({
                "dataset_split": "train[:1500]",
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "max_seq_length": 512,
                "fp16": True,
            })
        elif "v100" in gpu_name:
            print("⚙️  Optimizing for V100 — estimated training time: ~30-35 min")
            base_config.update({
                "dataset_split": "train[:1200]",
                "per_device_train_batch_size": 6,
                "gradient_accumulation_steps": 3,
                "max_seq_length": 512,
                "fp16": True,
            })
        elif "a100" in gpu_name:
            print("⚙️  Optimizing for A100 — estimated training time: ~15-20 min")
            base_config.update({
                "dataset_split": "train[:2000]",
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "max_seq_length": 512,
                "fp16": True,
            })
        elif "h100" in gpu_name:
            print("⚙️  Optimizing for H100 — estimated training time: ~10-15 min")
            base_config.update({
                "dataset_split": "train[:3000]",
                "per_device_train_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "max_seq_length": 512,
                "fp16": True,
                "num_train_epochs": 4,
            })
        else:
            print(f"⚙️  Unknown GPU — scaling conservatively by VRAM ({gpu_memory:.1f} GB)")
            if gpu_memory >= 40:
                batch_size, grad_accum = 16, 1
            elif gpu_memory >= 24:
                batch_size, grad_accum = 8, 2
            elif gpu_memory >= 16:
                batch_size, grad_accum = 4, 4
            else:
                batch_size, grad_accum = 2, 8

            base_config.update({
                "dataset_split": "train[:1000]",
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "max_seq_length": 512,
                "fp16": True,
            })

    # Apply any caller-provided overrides last
    if config_overrides:
        base_config.update(config_overrides)

    return base_config


# ─────────────────────────────────────────────
# 2. Layer Collapse (LaCo)
# ─────────────────────────────────────────────

class LayerCollapser:
    """Implements LaCo (Layer Collapse) technique for model pruning."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _get_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError("Could not find transformer layers in the model.")

    @staticmethod
    def _merge_two_layers(layer1: nn.Module, layer2: nn.Module) -> nn.Module:
        """Average the weights of two layers and store result in layer1."""
        sd1 = layer1.state_dict()
        sd2 = layer2.state_dict()
        merged = {k: (sd1[k] + sd2[k]) / 2.0 for k in sd1 if k in sd2}
        for k in sd1:
            if k not in merged:
                merged[k] = sd1[k]
        layer1.load_state_dict(merged)
        return layer1

    def collapse_layers(self, layers_to_merge: List[Tuple[int, int]]):
        """Merge specified adjacent layer pairs in-place."""
        layers_list = list(self._get_layers())
        original_count = len(layers_list)
        print(f"Original model: {original_count} layers")
        print(f"Merging pairs: {layers_to_merge}")

        for idx1, idx2 in sorted(layers_to_merge, reverse=True):
            if idx2 >= len(layers_list) or idx1 >= len(layers_list):
                print(f"  ⚠️  Skipping ({idx1}, {idx2}) — out of range")
                continue
            if idx2 != idx1 + 1:
                print(f"  ⚠️  Layers {idx1} and {idx2} are not adjacent")
            print(f"  Merging layers {idx1} + {idx2}...")
            layers_list[idx1] = self._merge_two_layers(layers_list[idx1], layers_list[idx2])
            layers_list.pop(idx2)

        self.model.model.layers = nn.ModuleList(layers_list)
        self.model.config.num_hidden_layers = len(layers_list)

        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = i

        if hasattr(self.model.config, "layer_types") and self.model.config.layer_types is not None:
            layer_types = list(self.model.config.layer_types)
            for idx1, idx2 in sorted(layers_to_merge, reverse=True):
                if idx2 < len(layer_types):
                    layer_types.pop(idx2)
            self.model.config.layer_types = layer_types
            print(f"  Updated config.layer_types: {len(layer_types)} entries")

        print(f"✓ Collapsed: {original_count} → {len(layers_list)} layers")
        print(f"  Updated config.num_hidden_layers = {len(layers_list)}")
        return self.model


# ─────────────────────────────────────────────
# 3. Helper Functions
# ─────────────────────────────────────────────

def load_and_prepare_dataset(tokenizer, config: dict):
    """Load and tokenize the Guanaco instruction dataset."""
    print("Loading dataset...")
    dataset = load_dataset(config["dataset"], split=config["dataset_split"])
    print(f"  {len(dataset)} examples loaded")

    def tokenize(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["max_seq_length"],
            padding="max_length",
            return_tensors=None,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return tokenized


def test_model(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """Run a quick inference check and print the result."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\nPrompt : {prompt}")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print("-" * 60)
    return response


def get_dir_size_gb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


# ─────────────────────────────────────────────
# 4. Main Exported Function
# ─────────────────────────────────────────────

def run_pruning_pipeline(
    model_path: str = "google/gemma-3-1b-it",
    hf_token: Optional[str] = None,
    output_dir: str = "./compressed_gemma",
    lora_output_dir: str = "./lora_adapters",
    final_merged_dir: str = "./final_healed_model",
    zip_output: str = "./compressed_gemma_final.zip",
    layers_to_merge: Optional[List[Tuple[int, int]]] = None,
    dataset: str = "timdettmers/openassistant-guanaco",
    config_overrides: Optional[dict] = None,
) -> dict:
    """
    End-to-end pruning + LoRA healing + merge pipeline.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local path.
    hf_token : str | None
        HuggingFace token. Falls back to ``HF_TOKEN`` env var.
    output_dir : str
        Directory for the pruned (collapsed) model.
    lora_output_dir : str
        Directory for LoRA adapter checkpoints.
    final_merged_dir : str
        Directory for the final merged model.
    zip_output : str
        Path for the output zip archive.
    layers_to_merge : list of (int, int) tuples | None
        Layer pairs to collapse. Defaults to [(2,3), (8,9)].
    dataset : str
        HuggingFace dataset ID for LoRA healing.
    config_overrides : dict | None
        Arbitrary overrides applied on top of auto-detected config.

    Returns
    -------
    dict with keys: ``pruned_model_dir``, ``final_model_dir``, ``zip_path``,
    ``original_layers``, ``final_layers``, ``pruned_size_gb``, ``final_size_gb``.
    """

    # ── HF Authentication ────────────────────────────────────────────────
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError(
            "HF token is empty.\n"
            "  Pass hf_token= or set the HF_TOKEN environment variable.\n"
        )
    login(token=token, add_to_git_credential=False)
    print("✓ Successfully logged in to Hugging Face!")

    # ── Build config ─────────────────────────────────────────────────────
    CONFIG = get_gpu_optimized_config(
        model_path=model_path,
        output_dir=output_dir,
        lora_output_dir=lora_output_dir,
        final_merged_dir=final_merged_dir,
        zip_output=zip_output,
        layers_to_merge=layers_to_merge,
        dataset=dataset,
        config_overrides=config_overrides,
    )

    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    for k, v in CONFIG.items():
        if k != "lora_target_modules":
            print(f"  {k:<35} {v}")
    print("=" * 70)

    # ── Phase 1: Layer Collapse ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: LAYER COLLAPSE (LaCo)")
    print("=" * 70)

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        dtype=torch.float16,
        device_map="auto",
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_layers = model.config.num_hidden_layers

    print("\n--- Inference BEFORE pruning ---")
    test_model(model, tokenizer, "What is machine learning?")

    print("\nApplying layer collapse...")
    collapser = LayerCollapser(model, tokenizer)
    model = collapser.collapse_layers(CONFIG["layers_to_merge"])

    final_layers = model.config.num_hidden_layers

    print("\n--- Inference AFTER pruning (degradation expected) ---")
    test_model(model, tokenizer, "What is machine learning?")

    print(f"\nSaving pruned model → {CONFIG['output_dir']}")
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print("✓ Phase 1 complete!\n")

    # ── Phase 2: LoRA Fine-tuning ────────────────────────────────────────
    print("=" * 70)
    print("PHASE 2: LORA FINE-TUNING (HEALING)")
    print("=" * 70)

    train_dataset = load_and_prepare_dataset(tokenizer, CONFIG)

    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["lora_target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=CONFIG["lora_output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        fp16=CONFIG.get("fp16", True),
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()

    print(f"\nSaving LoRA adapters → {CONFIG['lora_output_dir']}")
    model.save_pretrained(CONFIG["lora_output_dir"])

    print("\n--- Inference AFTER LoRA healing ---")
    test_model(model, tokenizer, "What is machine learning?")
    print("✓ Phase 2 complete!\n")

    # ── Phase 3: Merge and Save ──────────────────────────────────────────
    print("=" * 70)
    print("PHASE 3: MERGE LORA AND SAVE FINAL MODEL")
    print("=" * 70)

    print("\nLoading pruned base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["output_dir"],
        dtype=torch.float16,
        device_map="auto",
    )
    print("Loading LoRA adapters...")
    peft_model = PeftModel.from_pretrained(base_model, CONFIG["lora_output_dir"])

    print("Merging LoRA adapters into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving final healed model → {CONFIG['final_merged_dir']}")
    merged_model.save_pretrained(CONFIG["final_merged_dir"])
    tokenizer.save_pretrained(CONFIG["final_merged_dir"])

    print("\n--- Final Inference Tests ---")
    test_model(merged_model, tokenizer, "What is machine learning?")
    test_model(merged_model, tokenizer, "Explain neural networks in simple terms.")

    zip_base = CONFIG["zip_output"].replace(".zip", "")
    print(f"\nCreating zip archive → {CONFIG['zip_output']}")
    shutil.make_archive(zip_base, "zip", CONFIG["final_merged_dir"])
    print("✓ Phase 3 complete!\n")

    # ── Summary ──────────────────────────────────────────────────────────
    pruned_size = get_dir_size_gb(CONFIG["output_dir"]) if os.path.exists(CONFIG["output_dir"]) else 0.0
    final_size = get_dir_size_gb(CONFIG["final_merged_dir"]) if os.path.exists(CONFIG["final_merged_dir"]) else 0.0

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, path in [
        ("Pruned model", CONFIG["output_dir"]),
        ("Final healed model", CONFIG["final_merged_dir"]),
    ]:
        if os.path.exists(path):
            print(f"  {label}: {get_dir_size_gb(path):.2f} GB  ({path})")
    print(f"  Zip archive : {CONFIG['zip_output']}")
    print("\n✓ Pipeline completed successfully!")

    return {
        "pruned_model_dir": os.path.abspath(CONFIG["output_dir"]),
        "final_model_dir": os.path.abspath(CONFIG["final_merged_dir"]),
        "zip_path": os.path.abspath(CONFIG["zip_output"]),
        "original_layers": original_layers,
        "final_layers": final_layers,
        "pruned_size_gb": pruned_size,
        "final_size_gb": final_size,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pruning_pipeline()
