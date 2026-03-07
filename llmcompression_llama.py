"""
Knowledge Distillation Pipeline: Llama 3.1 8B (Teacher) -> Llama 3.2 1B (Student)

Extracts domain-specific knowledge from a PDF document using a teacher model,
generates synthetic Q&A training data, and fine-tunes a smaller student model
via LoRA + SFTTrainer. The final merged model is saved to disk.
"""

import os
import re
import gc
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_pdf_text(pdf_path: str) -> str:
    """Convert PDF pages to images and run OCR to extract text."""
    images = convert_from_path(pdf_path, dpi=200)
    full_text = ""
    for img in tqdm(images, desc="OCR processing"):
        try:
            full_text += pytesseract.image_to_string(img, lang="eng") + "\n"
        except Exception:
            continue

    full_text = re.sub(r"\s+", " ", full_text)
    full_text = re.sub(r"Page \d+", "", full_text)
    full_text = re.sub(r"\d+\s+Chapter", "", full_text)
    return full_text.strip()


def _chunk_text(text: str, chunk_size: int = 2000, min_chunk_size: int = 500) -> list:
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        if len(chunk) > min_chunk_size:
            chunks.append(chunk)
    return chunks


def _find_relevant_context(question: str, chunks: list, max_chunks: int = 5) -> str:
    keywords = question.lower().split()
    scored = []
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw in chunk.lower())
        scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    relevant = [chunk for score, chunk in scored[:max_chunks] if score > 0]
    return "\n".join(relevant) if relevant else chunks[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_knowledge_distillation(
    hf_token: str,
    pdf_path: str,
    output_dir: str,
    teacher_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    student_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    max_questions: int = 400,
    num_train_epochs: int = 15,
    seed: int = 42,
) -> dict:
    """Run the full knowledge-distillation pipeline.

    Steps
    -----
    1. Load teacher model (Llama 3.1 8B, 4-bit quantised).
    2. Extract text from the supplied PDF via OCR.
    3. Generate synthetic training Q&A pairs using the teacher.
    4. Fine-tune the student model (Llama 3.2 1B) with LoRA / SFTTrainer.
    5. Merge LoRA adapters back into the student and save.

    Parameters
    ----------
    hf_token : str
        Hugging Face access token (needs read access to Meta Llama models).
    pdf_path : str
        Path to a domain-specific PDF document used for training-data
        generation.
    output_dir : str
        Base directory for all intermediate and final outputs.
    teacher_model_name : str
        HF model id for the teacher.
    student_model_name : str
        HF model id for the student.
    max_questions : int
        Cap on the number of synthetic questions generated.
    num_train_epochs : int
        Number of training epochs for student fine-tuning.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``final_model_dir``  – path to the merged student model.
        ``training_data_path`` – path to the saved JSON training data.
        ``num_training_examples`` – number of Q&A pairs generated.
    """
    _set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    login(token=hf_token, add_to_git_credential=False)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ------------------------------------------------------------------
    # 1. Load teacher model
    # ------------------------------------------------------------------
    logger.info("Loading teacher model: %s", teacher_model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 2. Extract & chunk PDF content
    # ------------------------------------------------------------------
    logger.info("Extracting text from PDF: %s", pdf_path)
    full_text = _extract_pdf_text(pdf_path)
    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("No usable text chunks extracted from PDF")
    logger.info("Extracted %d text chunks", len(chunks))

    # ------------------------------------------------------------------
    # Helper: generate a single domain-aware teacher response
    # ------------------------------------------------------------------
    def _generate_domain_response(question: str, max_length: int = 400) -> str:
        context = _find_relevant_context(question, chunks, max_chunks=5)
        user_prompt = (
            "You are a domain expert. Use the following reference material "
            "to answer the question accurately and comprehensively.\n\n"
            f"Reference Material:\n{context[:1200]}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, accurate, and detailed answer based on the "
            "reference material."
        )
        messages = [{"role": "user", "content": user_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # 3. Synthesise training questions from PDF chunks
    # ------------------------------------------------------------------
    logger.info("Generating synthetic questions from PDF content...")
    synthetic_prompts: list[str] = []

    for chunk in tqdm(chunks, desc="Synthesising questions"):
        try:
            synthesis_prompt = (
                "You are an educational assistant. Based on the provided text, "
                "generate 3 diverse, high-quality questions that can be answered "
                "specifically using the information in the text.\n\n"
                f"Text Content:\n{chunk[:1500]}\n\n"
                "Output format:\n- Question 1\n- Question 2\n- Question 3\n\n"
                "Provide only the questions, one per line starting with a dash."
            )
            messages = [{"role": "user", "content": synthesis_prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=2048
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("-") or (len(line) > 0 and line[0].isdigit()):
                    clean_q = re.sub(r"^\d+[.\)]\s*|^-\s*", "", line).strip()
                    if len(clean_q) > 10:
                        synthetic_prompts.append(clean_q)
        except Exception:
            continue

    domain_prompts = list(set(synthetic_prompts))
    if len(domain_prompts) > max_questions:
        domain_prompts = random.sample(domain_prompts, max_questions)
    logger.info("Generated %d unique questions", len(domain_prompts))

    # ------------------------------------------------------------------
    # 4. Generate teacher responses (knowledge distillation data)
    # ------------------------------------------------------------------
    logger.info("Generating teacher responses...")
    training_data: list[dict] = []

    for question in tqdm(domain_prompts, desc="Generating responses"):
        try:
            answer = _generate_domain_response(question)
            if not answer.startswith("[Error:"):
                training_data.append(
                    {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    }
                )
        except Exception:
            continue

    logger.info("Generated %d training conversations", len(training_data))

    training_data_path = os.path.join(output_dir, "domain_specific_chat_data.json")
    with open(training_data_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Release teacher model
    # ------------------------------------------------------------------
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # 5. Fine-tune student model
    # ------------------------------------------------------------------
    logger.info("Loading student model: %s", student_model_name)

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, token=True)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
    )
    student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "right"

    def _format_conversation(example):
        formatted = student_tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": formatted}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(_format_conversation)
    split = dataset.train_test_split(test_size=0.1, seed=seed)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_output_dir = os.path.join(output_dir, "llama-chat-domain")

    training_args = TrainingArguments(
        output_dir=lora_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=student_model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=lora_config,
        args=training_args,
    )

    logger.info("Starting student fine-tuning (%d epochs)...", num_train_epochs)
    trainer.train()

    adapter_path = os.path.join(output_dir, "llama-chat-final")
    trainer.model.save_pretrained(adapter_path)
    student_tokenizer.save_pretrained(adapter_path)

    # ------------------------------------------------------------------
    # 6. Merge LoRA adapters into the student base model
    # ------------------------------------------------------------------
    del student_model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Merging LoRA adapters into student base model...")
    merge_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=True,
    )
    merge_model = PeftModel.from_pretrained(merge_model, adapter_path)
    merged_model = merge_model.merge_and_unload()

    final_model_dir = os.path.join(output_dir, "llama-domain-specialist-merged")
    merged_model.save_pretrained(final_model_dir)

    final_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    final_tokenizer.save_pretrained(final_model_dir)

    del merged_model
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Knowledge distillation pipeline complete. Output: %s", final_model_dir)

    return {
        "final_model_dir": final_model_dir,
        "training_data_path": training_data_path,
        "num_training_examples": len(training_data),
    }
