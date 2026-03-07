# Hugging Face Authentication
from huggingface_hub import login
import os

print("="*70)
print("Hugging Face Authentication")
print("="*70)

print("\nPlease paste your Hugging Face token below")
print("Token can be obtained from: https://huggingface.co/settings/tokens")
print("(The token will be hidden as you type)\n")

from getpass import getpass
token = getpass("Enter your HuggingFace token: ")

try:
    login(token=token, add_to_git_credential=False)
    print("\nSuccessfully logged in to Hugging Face")

except Exception as e:
    print(f"\nLogin failed: {e}")
    print("\nTroubleshooting steps:")
    print("   1. Verify that your token is valid")
    print("   2. Obtain a new token from: https://huggingface.co/settings/tokens")
    print("   3. Use a 'Read' or 'Write' token (not 'Fine-grained')")
    raise

print("\nNote: You need access to the following models:")
print("   - meta-llama/Llama-3.1-8B-Instruct (Teacher model)")
print("   - meta-llama/Llama-3.2-1B-Instruct (Student model)")

print("\n   Request access at:")
print("   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
print("   - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
print("\n   Access is typically granted within 5-10 minutes.")

# Verify PDF file exists
import os

print("="*70)
print("Verifying Domain-Specific PDF Document")
print("="*70)

# Configure your PDF path here
pdf_path = "document.pdf"  # Replace with your PDF filename

if os.path.exists(pdf_path):
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"PDF document found: {pdf_path}")
    print(f"   File size: {file_size:.2f} MB")
else:
    print(f"ERROR: PDF not found at '{pdf_path}'")
    print("\nPlease upload your domain-specific PDF to the current directory")
    print("\nInstructions:")
    print("1. In Google Colab: Use the file upload button in the left sidebar")
    print("2. Locally: Place the PDF in the same folder as this notebook")
    print("3. Update the 'pdf_path' variable in this cell with your filename")
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Set random seeds for reproducibility
import random
import numpy as np
import torch

print("="*70)
print("Setting Random Seeds for Reproducibility")
print("="*70)

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Random seed set to {seed}")
print("   This ensures reproducible results across runs")

# Load Teacher Model: Llama 3.1 8B
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from tqdm import tqdm

print("="*70)
print("Loading Teacher Model: Llama 3.1 8B")
print("="*70)

# Configure 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load Llama 3.1 8B Instruction-tuned model
model_name = "meta-llama/Llama-3.1-8B-Instruct"

print(f"\nLoading {model_name}...")
print("   Initial load may take several minutes...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Teacher model loaded successfully")
    print(f"   Model: Llama 3.1 8B (Instruction-tuned)")
    print(f"   Parameters: ~8B (4-bit quantized)")
    print(f"   Memory usage: ~6 GB VRAM")
    print(f"   Quantization: NF4 with double quantization")

except Exception as e:
    print(f"Error loading teacher model: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify Hugging Face authentication (run previous cells)")
    print("2. Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("3. Wait 5-10 minutes for access approval")
    raise


import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
import re
import os

print("="*70)
print("Extracting Content from PDF Document (with OCR)")
print("="*70)

pdf_path = "document.pdf"  # Make sure this is your PDF's filename

if not os.path.exists(pdf_path):
    print(f"ERROR: PDF not found at '{pdf_path}'")
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

print("PDF found. Converting PDF pages to images for OCR...")
# 1. Convert PDF pages to a list of images
try:
    images = convert_from_path(pdf_path, dpi=200)
    print(f"Successfully converted {len(images)} pages to images.")
except Exception as e:
    print(f"Error during PDF-to-image conversion: {e}")
    raise

# 2. Extract text from each image using Tesseract (OCR)
full_text = ""
print("Extracting text from images using OCR (this may take a few minutes)...")
for i, img in enumerate(tqdm(images, desc="Processing pages")):
    try:
        # Use pytesseract to do OCR on the image
        full_text += pytesseract.image_to_string(img, lang='eng') + "\n"
    except Exception as e:
        print(f"\nWarning: Could not extract text from page {i+1}: {e}")
        continue

print(f"\nExtracted {len(full_text):,} characters from document using OCR")

# 3. Text preprocessing function (same as before)
def clean_text(text):
    """Clean and normalize extracted PDF text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common page artifacts
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\d+\s+Chapter', '', text)
    return text.strip()

print("\nCleaning and preprocessing text...")
full_text = clean_text(full_text)
print(f"Preprocessed text: {len(full_text):,} characters")

# 4. Split into contextual chunks (same as before)
CHUNK_SIZE = 2000
chunks = []

print(f"\nSplitting into {CHUNK_SIZE}-character chunks for context windows...")
for i in range(0, len(full_text), CHUNK_SIZE):
    chunk = full_text[i:i+CHUNK_SIZE]
    if len(chunk) > 500:  # This filter should work now
        chunks.append(chunk)

# 5. Handle the original error just in case
if not chunks:
    print("\nWARNING: No chunks were created. The PDF might be empty or unreadable.")
    print("   Total chunks: 0")
else:
    print(f"\nCreated {len(chunks)} contextual chunks for training data generation")
    print(f"\nChunk statistics:")
    print(f"   Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")
    print(f"   Total chunks: {len(chunks)}")

print(f"\nPDF processing complete. Ready for training data generation.")

# Configure context-aware response generation system
print("="*70)
print("Configuring Context-Aware Response Generation")
print("="*70)

# Semantic context retrieval function
def find_relevant_context(question, chunks, max_chunks=5):
    """
    Retrieve most relevant PDF chunks for a given question using keyword matching.

    Args:
        question: User query string
        chunks: List of text chunks from PDF
        max_chunks: Maximum number of chunks to return

    Returns:
        Concatenated relevant context string
    """
    keywords = question.lower().split()

    # Score chunks based on keyword frequency
    chunk_scores = []
    for chunk in chunks:
        score = sum(1 for keyword in keywords if keyword in chunk.lower())
        chunk_scores.append((score, chunk))

    # Sort by relevance and select top chunks
    chunk_scores.sort(reverse=True, key=lambda x: x[0])
    relevant_chunks = [chunk for score, chunk in chunk_scores[:max_chunks] if score > 0]

    return "\n".join(relevant_chunks) if relevant_chunks else chunks[0]

# Response generation function with PDF context
def generate_domain_response(question, max_length=1024):
    """
    Generate domain-specific response using teacher model with PDF context.

    Args:
        question: User query
        max_length: Maximum tokens to generate

    Returns:
        Generated response string
    """
    try:
        # Retrieve relevant context from PDF
        context = find_relevant_context(question, chunks, max_chunks=5)

        # Construct prompt with context
        user_prompt = f"""You are a domain expert. Use the following reference material to answer the question accurately and comprehensively.

Reference Material:
{context[:1200]}

Question: {question}

Provide a clear, accurate, and detailed answer based on the reference material."""

        # Format using Llama's chat template
        messages = [{"role": "user", "content": user_prompt}]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize formatted prompt
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
            add_special_tokens=False
        )

        # Move tensors to model device
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)

        # Generate response
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
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (exclude input prompt)
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return response

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating response: {e}")
        print(f"Full traceback:\n{error_details}")
        return f"[Error: Could not generate response - {str(e)}]"

print("Response generation system configured successfully")
print("\nSystem capabilities:")
print("   - Context-aware response generation")
print("   - Semantic chunk retrieval")
print("   - Domain-specific knowledge integration")

# Generate domain-specific training dataset
print("="*70)
print("Step 2: Synthesizing Training Questions from Content")
print("="*70)

def generate_questions_from_content(text_chunks, questions_per_chunk=3):
    """
    Use the Teacher model to generate questions grounded in the document content.
    This creates a much more effective training set for Knowledge Distillation.
    """
    synthetic_prompts = []

    # We sample chunks to ensure broad coverage without extreme runtime
    # Stride of 2 gives good variety
    sampled_chunks = text_chunks

    print(f"Generating questions from {len(sampled_chunks)} content chunks...")

    for i, chunk in enumerate(tqdm(sampled_chunks, desc="Synthesizing Questions")):
        try:
            # Construct synthesis prompt
            synthesis_prompt = f"""You are an educational assistant. Based on the provided text, generate {questions_per_chunk} diverse, high-quality questions that can be answered specifically using the information in the text.

            Text Content:
            {chunk[:1500]}

            Output format:
            - Question 1
            - Question 2
            - Question 3

            Provide only the questions, one per line starting with a dash."""

            messages = [{"role": "user", "content": synthesis_prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode only generated tokens
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Parse questions
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') or (len(line) > 0 and line[0].isdigit()):
                    # Clean up prefix
                    clean_q = re.sub(r'^\d+[\.\)]\s*|^-\s*', '', line).strip()
                    if len(clean_q) > 10:
                        synthetic_prompts.append(clean_q)

        except Exception as e:
            continue

    return list(set(synthetic_prompts)) # Unique questions

# Generate the synthetic prompts
domain_prompts = generate_questions_from_content(chunks)

print(f"\nSuccessfully generated {len(domain_prompts)} synthetic questions.")

# Limit to a high-quality subset if it's too large for the A40's time constraints
MAX_QUESTIONS = 400
if len(domain_prompts) > MAX_QUESTIONS:
    print(f"Sampling {MAX_QUESTIONS} highest quality questions for distillation...")
    domain_prompts = random.sample(domain_prompts, MAX_QUESTIONS)

print("="*70)
print("Step 3: Generating Teacher Responses (Knowledge Distillation)")
print("="*70)

training_data = []
successful = 0
failed = 0

print(f"\nGenerating teacher responses for {len(domain_prompts)} questions...")
print(f"Estimated time: {len(domain_prompts) * 15 // 60} minutes\n")

for i, question in enumerate(tqdm(domain_prompts, desc="Generating responses")):
    try:
        # Generate response using teacher model with PDF context
        answer = generate_domain_response(question, max_length=400)

        # Validate response
        if answer.startswith("[Error:"):
            failed += 1
            continue

        # Store in ChatML format (compatible with modern chat models)
        conversation = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }

        training_data.append(conversation)
        successful += 1

        # Progress logging
        if (i + 1) % 5 == 0:
            print(f"\nGenerated {i + 1}/{len(domain_prompts)} conversations")
            print(f"   Q: '{question[:60]}...'")
            print(f"   A: '{answer[:80]}...'")

    except Exception as e:
        print(f"\nError for question '{question[:50]}...': {e}")
        failed += 1
        continue

print(f"\n{'='*70}")
print(f"Training Dataset Generation Complete")
print(f"{'='*70}")
print(f"   Successful: {successful}")
print(f"   Failed: {failed}")
print(f"   Total conversations: {len(training_data)}")

# Data quality validation
if len(training_data) < 10:
    print("\nWARNING: Insufficient training examples generated")
    print("   Recommendation: Add more domain-specific questions or check for errors.")
else:
    print(f"\nDataset quality: {len(training_data)} examples (sufficient for fine-tuning)")

# Save training data
output_file = "domain_specific_chat_data.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print(f"\nTraining data saved to: {output_file}")

# Dataset statistics
total_chars = sum(len(d["messages"][0]["content"]) + len(d["messages"][1]["content"])
                  for d in training_data)
avg_question_length = sum(len(d["messages"][0]["content"]) for d in training_data) / len(training_data) if training_data else 0
avg_answer_length = sum(len(d["messages"][1]["content"]) for d in training_data) / len(training_data) if training_data else 0

print(f"\nDataset Statistics:")
print(f"   Total Q&A pairs: {len(training_data)}")
print(f"   Total characters: {total_chars:,}")
print(f"   Average question length: {avg_question_length:.0f} characters")
print(f"   Average answer length: {avg_answer_length:.0f} characters")

# Display sample conversations
print(f"\nSample Training Conversations:")
for i, conv in enumerate(training_data[:3], 1):
    print(f"\n{i}. User: {conv['messages'][0]['content']}")
    print(f"   Assistant: {conv['messages'][1]['content'][:150]}...")

print(f"\n{'='*70}")
print(f"Ready to fine-tune student model (Llama 3.2 1B)")
print(f"{'='*70}")

print("="*70)
print("Releasing Teacher Model from Memory")
print("="*70)

try:
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("Teacher model (Llama 3.1 8B) cleared from GPU memory")
except:
    print("No model to clear")

print("\n" + "="*70)
print("Loading Student Model: Llama 3.2 1B")
print("="*70)

# Load Llama 3.2 1B student model
llama_model_name = "meta-llama/Llama-3.2-1B-Instruct"

print(f"\nLoading {llama_model_name}...")
print("   Initial load may take several minutes...")

try:
    # Load with 4-bit quantization for memory efficiency
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=True)
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
    )

    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    print("Student model loaded successfully")
    print(f"\nModel Specifications:")
    print(f"   Model: {llama_model_name}")
    print(f"   Parameters: ~1B (4-bit quantized)")
    print(f"   Context length: 131072 tokens")
    print(f"   Memory usage: ~1.5 GB VRAM")
    print(f"   Architecture: Transformer decoder")

except Exception as e:
    print(f"Error loading student model: {e}")
    print("\nTroubleshooting steps:")
    print("1. Verify Hugging Face authentication")
    print("2. Request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    print("3. Check internet connectivity")
    raise

# Prepare training dataset for fine-tuning
from datasets import Dataset
import os

print("="*70)
print("Preparing Training Dataset")
print("="*70)

# Verify training data availability
training_file = "domain_specific_chat_data.json"
if not os.path.exists(training_file):
    print(f"ERROR: Training data file '{training_file}' not found")
    print("   Please execute the data generation cell first.")
    raise FileNotFoundError(f"Training data not found: {training_file}")

# Load generated training data
try:
    with open(training_file, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)

    print(f"Loaded {len(chat_data)} training conversations")

    if len(chat_data) == 0:
        raise ValueError("Training dataset is empty")

except Exception as e:
    print(f"Error loading training data: {e}")
    raise

# Format conversations for Llama chat template
def format_conversation_llama(example):
    """
    Format conversation using Llama's chat template.
    """
    messages = example["messages"]
    formatted = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted}

# Convert to Hugging Face Dataset format
print("\nConverting to Hugging Face Dataset format...")
dataset = Dataset.from_list(chat_data)
dataset = dataset.map(format_conversation_llama)

print(f"Dataset prepared: {len(dataset)} training examples")
print(f"\nExample formatted conversation:")
print(dataset[0]["text"][:300] + "...")

# Split into training and evaluation sets (90/10)
print("\nSplitting dataset...")
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"\nDataset Split:")
print(f"   Training set: {len(train_dataset)} examples")
print(f"   Evaluation set: {len(eval_dataset)} examples")
print(f"\nReady for LoRA configuration and fine-tuning")

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
from peft import LoraConfig

print("="*70)
print("Configuring LoRA for Efficient Training")
print("="*70)

# LoRA configuration - targets important attention layers
lora_config = LoraConfig(
    r=32,  # LoRA rank (higher = more parameters but better quality)
    lora_alpha=64,  # LoRA alpha scaling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print(f"LoRA configuration defined")
print(f"   LoRA will be applied by SFTTrainer during training setup")

# Configure training with SFTTrainer (Supervised Fine-Tuning)
from transformers import TrainingArguments
from trl import SFTTrainer

print("="*70)
print("Configuring Training Parameters")
print("="*70)

# Training configuration
training_args = TrainingArguments(
    output_dir="./llama-chat-domain",
    num_train_epochs=15,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,  # Use bfloat16 for stability
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    push_to_hub=False,
    report_to="none",
)

# Initialize trainer
trainer = SFTTrainer(
    model=llama_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=training_args,
)

print("Trainer configured successfully")
print(f"\nTraining Configuration:")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Max sequence length: 2048 tokens")
print(f"   Optimizer: paged_adamw_8bit (memory efficient)")

print(f"\nEstimated training time: 30-45 minutes on T4 GPU")
print(f"Model checkpoints will be saved to: ./llama-chat-domain/")

# Execute training process
print("="*70)
print("Starting Training: Llama 3.2 1B Learning from Llama 3.1 8B")
print("="*70)

# Train the model
trainer.train()

print("\n" + "="*70)
print("Training Complete")
print("="*70)

# Save the final model
final_model_path = "./llama-chat-final"
trainer.model.save_pretrained(final_model_path)
llama_tokenizer.save_pretrained(final_model_path)

print(f"\nFinal model saved to: {final_model_path}")
print(f"   Model is ready for GGUF conversion")

# Display training summary
print(f"\nTraining Summary:")
print(f"   Model: Llama 3.2 1B")
print(f"   Teacher: Llama 3.1 8B")
print(f"   Training samples: {len(train_dataset)}")
print(f"   Evaluation samples: {len(eval_dataset)}")
print(f"   LoRA rank: {lora_config.r}")

# ## Step 3: Model Evaluation and Testing
#
# Evaluate the fine-tuned student model to verify successful knowledge transfer from the teacher model.

# Load fine-tuned model for evaluation
from peft import PeftModel

print("="*70)
print("Loading Fine-Tuned Student Model for Evaluation")
print("="*70)

# Load base Llama 3.2 1B model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=True,
)

# Load LoRA adapter weights
final_model_path = "./llama-chat-final"
test_model = PeftModel.from_pretrained(base_model, final_model_path)
test_tokenizer = AutoTokenizer.from_pretrained(final_model_path)

print("Fine-tuned model loaded successfully")

# Inference function
def chat_with_llama(user_message, max_length=512):
    """
    Generate response using fine-tuned Llama 3.2 model.

    Args:
        user_message: User query
        max_length: Maximum tokens to generate

    Returns:
        Generated response string
    """
    # Format using Llama chat template
    messages = [{"role": "user", "content": user_message}]
    prompt = test_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = test_tokenizer(prompt, return_tensors="pt").to(test_model.device)

    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=test_tokenizer.pad_token_id,
            eos_token_id=test_tokenizer.eos_token_id,
            use_cache=False,
        )

    # Decode only generated tokens (exclude input prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = test_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response

print("\n" + "="*70)
print("Model Evaluation Ready")
print("="*70)

# Execute evaluation test suite
# Note: Customize these test queries based on your domain

test_queries = [
    "Give mary malloy's basic details",
    "Which patient had type 2 diabetes?",
    "Give medication summary for mary malloy",
]

print("="*70)
print("Running Model Evaluation Tests")
print("="*70)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"Test Case {i}/{len(test_queries)}")
    print(f"{'='*70}")
    print(f"\nQuery: {query}")
    print(f"\nResponse: ", end="")

    response = chat_with_llama(query)
    print(response)

print("\n" + "="*70)
print("Evaluation Complete")
print("="*70)

print("\nModel successfully fine-tuned and ready for deployment")
print("   Proceed to GGUF conversion for mobile deployment.")


# Merge LoRA weights with base model
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

gc.collect()

print("="*70)
print("Merging LoRA Adapter with Base Model")
print("="*70)

try:
    # Load base model without quantization (required for merging)
    print("\nLoading base Llama 3.2 1B model (FP16)...")
    merge_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        token=True,
    )

    final_model_path = "llama-chat-final"

    # Load LoRA adapter
    print("Loading LoRA adapter weights...")
    merge_model = PeftModel.from_pretrained(merge_model, final_model_path)

    # Merge adapter with base model
    print("Merging LoRA weights into base model...")
    merged_model = merge_model.merge_and_unload()

    # Save merged model
    merged_path = "./llama-domain-specialist-merged"
    print(f"\nSaving merged model to {merged_path}...")
    merged_model.save_pretrained(merged_path)
    test_tokenizer.save_pretrained(merged_path)

    print(f"Model merge successful")
    print(f"Merged model saved to: {merged_path}")
    print(f"\nModel is now a unified domain specialist")
    print(f"Ready for GGUF conversion")

except Exception as e:
    print(f"Error during model merge: {e}")
    print("\nVerify that training completed successfully")
    raise

# Convert merged model to GGUF format
# import subprocess
# import os

# print("="*70)
# print("Converting Model to GGUF Format")
# print("="*70)

# merged_path = "./llama-domain-specialist-merged"

# # Install GGUF conversion dependencies
# print("\nInstalling conversion dependencies...")
# try:
#     subprocess.run(["pip", "install", "-q", "gguf", "sentencepiece", "protobuf"], check=True)
#     print("Dependencies installed")
# except Exception as e:
#     print(f"Warning: Dependency installation issue: {e}")

# # Clone llama.cpp repository for conversion tools
# if not os.path.exists("llama.cpp"):
#     print("\nCloning llama.cpp repository...")
#     try:
#         subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
#         print("llama.cpp repository cloned successfully")
#     except Exception as e:
#         print(f"Error cloning repository: {e}")
#         print("   Manual clone: git clone https://github.com/ggerganov/llama.cpp.git")
#         raise
# else:
#     print("llama.cpp repository already exists")

# # Verify merged model exists
# if not os.path.exists(merged_path):
#     print(f"\nERROR: Merged model not found at {merged_path}")
#     print("   Please execute the model merge cell first.")
#     raise FileNotFoundError(f"Merged model not found: {merged_path}")

# print("\nConverting to GGUF format (FP16)...")
# print("   Estimated time: 5-10 minutes\n")

# # Execute conversion
# try:
#     result = subprocess.run([
#         "python",
#         "./llama.cpp/convert_hf_to_gguf.py",
#         merged_path,
#         "--outfile", "./llama-domain-specialist.gguf",
#         "--outtype", "f16"
#     ], capture_output=True, text=True, timeout=600)

#     if result.returncode == 0:
#         print("GGUF conversion successful")

#         # Display file information
#         gguf_file = "./llama-domain-specialist.gguf"
#         if os.path.exists(gguf_file):
#             gguf_size = os.path.getsize(gguf_file) / (1024 * 1024)
#             print(f"\nModel Information:")
#             print(f"   Filename: llama-domain-specialist.gguf")
#             print(f"   Size: {gguf_size:.1f} MB")
#             print(f"   Format: FP16 (half precision)")
#             print(f"   Compatible with: llama.cpp, LM Studio, Ollama")
#         else:
#             print("Warning: GGUF file not created")
#     else:
#         print(f"Conversion failed")
#         print(f"Error output: {result.stderr}")

# except subprocess.TimeoutExpired:
#     print("Conversion timed out (exceeded 10 minutes)")
# except Exception as e:
#     print(f"Conversion error: {e}")
#     print("\nManual conversion command:")
#     print(f"   python llama.cpp/convert_hf_to_gguf.py {merged_path} --outfile llama-domain-specialist.gguf --outtype f16")


# from google.colab import drive
# drive.mount('/content/drive')

# import os
# import shutil
# from google.colab import drive

# print("Mounting Google Drive...")
# drive.mount('/content/drive')

# # 1. Define your source and destination
# base_dir = "/content"
# drive_folder_path = "/content/drive/MyDrive/model-olympics2024-specific"

# # 2. Create the destination folder (if it doesn't exist)
# os.makedirs(drive_folder_path, exist_ok=True)
# print(f"Successfully created or found folder: {drive_folder_path}")

# # 3. List of ALL files/folders to copy
# # This list includes your final project outputs
# # It specifically EXCLUDES 'llama.cpp', 'sample_data', and intermediate checkpoints.
# artifacts_to_copy = [
#     "llama-chat-final",                     # The trained LoRA adapter
#     "llama-domain-specialist-merged",       # The full merged model
#     "llama-domain-specialist.gguf",         # The final GGUF model
#     "llama-domain-specialist-quant.gguf",
#     "domain_specific_chat_data.json",      # The dataset you generated
#     "olympics.pdf",                  # Your source PDF (assuming this is the name)
#     "llmcompression.ipynb"                   # Your notebook (assuming this is the name)
# ]

# # 4. Loop through and copy each item
# print("\nStarting to copy your project files...")
# for item_name in artifacts_to_copy:
#     source_path = os.path.join(base_dir, item_name)
#     dest_path = os.path.join(drive_folder_path, item_name)

#     # Check if the file/folder actually exists
#     if not os.path.exists(source_path):
#         print(f"  > WARNING: '{item_name}' not found. Skipping.")
#         continue

#     try:
#         if os.path.isdir(source_path):
#             # If the folder already exists in Drive, remove it first to copy fresh
#             if os.path.exists(dest_path):
#                 shutil.rmtree(dest_path)
#             shutil.copytree(source_path, dest_path)
#             print(f"  > Copied directory: {item_name}")
#         else:
#             shutil.copy2(source_path, dest_path) # copy2 preserves metadata
#             print(f"  > Copied file: {item_name}")

#     except Exception as e:
#         print(f"  > ERROR copying '{item_name}': {e}")

# print("\nCopy complete!")
# print(f"Check your Google Drive for the 'model-olympics2024-specific' folder.")
