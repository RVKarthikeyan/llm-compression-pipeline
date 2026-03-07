"""
Pruning + LoRA Healing Pipeline for Llama-3.2-1B-Instruct

Two-stage model compression:
  Phase 1 – LaCo (Layer Collapse): merge adjacent transformer layers by
            averaging weights to reduce model depth.
  Phase 2 – LoRA Healing: fine-tune the pruned model with LoRA adapters on
            an instruction-following dataset to recover accuracy.
  Phase 3 – Merge LoRA adapters back into the base model and save.
"""

import os
import gc
import logging
import shutil
from typing import List, Tuple

import torch
import torch.nn as nn
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer Collapse implementation
# ---------------------------------------------------------------------------

class LayerCollapser:
    """Implements LaCo (Layer Collapse) for architecture pruning."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError("Could not find model layers")

    def merge_two_layers(self, layer1, layer2):
        sd1 = layer1.state_dict()
        sd2 = layer2.state_dict()
        merged = {}
        for key in sd1:
            merged[key] = (sd1[key] + sd2[key]) / 2.0 if key in sd2 else sd1[key]
        layer1.load_state_dict(merged)
        return layer1

    def collapse_layers(self, layers_to_merge: List[Tuple[int, int]]):
        layers = self.get_layers()
        original_count = len(layers)
        layers_list = list(layers)

        for idx1, idx2 in sorted(layers_to_merge, reverse=True):
            if idx2 >= len(layers_list) or idx1 >= len(layers_list):
                logger.warning("Cannot merge layers %d and %d – out of range", idx1, idx2)
                continue
            logger.info("Merging layers %d and %d ...", idx1, idx2)
            layers_list[idx1] = self.merge_two_layers(layers_list[idx1], layers_list[idx2])
            layers_list.pop(idx2)

        new_layers = nn.ModuleList(layers_list)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.layers = new_layers

        # Fix self_attn.layer_idx so KV-cache indices match new positions
        for new_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = new_idx

        logger.info("Collapsed from %d to %d layers", original_count, len(new_layers))
        return self.model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pruning_pipeline(
    hf_token: str,
    output_dir: str,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    layers_to_merge: List[Tuple[int, int]] | None = None,
    dataset_name: str = "timdettmers/openassistant-guanaco",
    dataset_split: str = "train[:1000]",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 256,
    learning_rate: float = 2e-4,
) -> dict:
    """Run the full pruning + LoRA-healing pipeline.

    Steps
    -----
    1. Load the base model.
    2. Apply LaCo layer collapse (merge specified adjacent layer pairs).
    3. Fine-tune the pruned model with LoRA on an instruction dataset.
    4. Merge LoRA adapters back into the pruned base model.
    5. Save the final healed model and a zip archive.

    Parameters
    ----------
    hf_token : str
        Hugging Face access token.
    output_dir : str
        Base directory for all intermediate and final outputs.
    model_name : str
        HF model id for the model to prune.
    layers_to_merge : list of (int, int) or None
        Adjacent layer pairs to collapse.  Defaults to ``[(2, 3), (7, 8)]``.
    dataset_name : str
        HF dataset id used for LoRA healing.
    dataset_split : str
        Dataset split string (e.g. ``"train[:1000]"``).
    lora_r, lora_alpha, lora_dropout : LoRA hyper-parameters.
    num_train_epochs : int
        Training epochs for LoRA healing.
    per_device_train_batch_size, gradient_accumulation_steps : int
        Training batch configuration.
    max_seq_length : int
        Maximum tokenised sequence length.
    learning_rate : float
        Peak learning rate for the cosine schedule.

    Returns
    -------
    dict
        ``final_model_dir``        – path to the merged healed model.
        ``pruned_model_dir``       – path to the intermediate pruned model.
        ``zip_path``               – path to a zip archive of the final model.
        ``num_layers_original``    – original layer count.
        ``num_layers_pruned``      – layer count after collapse.
    """
    if layers_to_merge is None:
        layers_to_merge = [(2, 3), (7, 8)]

    os.makedirs(output_dir, exist_ok=True)
    login(token=hf_token, add_to_git_credential=False)

    pruned_dir = os.path.join(output_dir, "compressed_llama")
    lora_dir = os.path.join(output_dir, "lora_adapters")
    final_dir = os.path.join(output_dir, "final_healed_model")

    # ------------------------------------------------------------------
    # Phase 1: Layer Collapse
    # ------------------------------------------------------------------
    logger.info("Phase 1 – Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_layers = len(model.model.layers)

    logger.info("Applying layer collapse: %s", layers_to_merge)
    collapser = LayerCollapser(model, tokenizer)
    model = collapser.collapse_layers(layers_to_merge)
    model.config.num_hidden_layers = len(model.model.layers)
    pruned_layers = len(model.model.layers)

    torch.cuda.empty_cache()

    logger.info("Saving pruned model to %s", pruned_dir)
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)

    # ------------------------------------------------------------------
    # Phase 2: LoRA Healing
    # ------------------------------------------------------------------
    logger.info("Phase 2 – LoRA fine-tuning for healing...")

    dataset = load_dataset(dataset_name, split=dataset_split)

    def _format_prompt(example):
        return {"text": example["text"]}

    def _tokenize(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    dataset = dataset.map(_format_prompt, remove_columns=dataset.column_names)
    tokenized = dataset.map(
        _tokenize, batched=True, remove_columns=dataset.column_names
    )

    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=lora_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_steps=100,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("Starting LoRA healing (%d epochs)...", num_train_epochs)
    trainer.train()

    logger.info("Saving LoRA adapters to %s", lora_dir)
    model.save_pretrained(lora_dir)

    # ------------------------------------------------------------------
    # Phase 3: Merge LoRA & save final model
    # ------------------------------------------------------------------
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Phase 3 – Merging LoRA adapters into pruned base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        pruned_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model_with_lora = PeftModel.from_pretrained(base_model, lora_dir)
    merged = model_with_lora.merge_and_unload()

    merged.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Create a convenience zip archive
    zip_base = os.path.join(output_dir, "compressed_llama_final")
    shutil.make_archive(zip_base, "zip", final_dir)
    zip_path = zip_base + ".zip"

    del merged
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Pruning pipeline complete. Output: %s", final_dir)

    return {
        "final_model_dir": final_dir,
        "pruned_model_dir": pruned_dir,
        "zip_path": zip_path,
        "num_layers_original": original_layers,
        "num_layers_pruned": pruned_layers,
    }
