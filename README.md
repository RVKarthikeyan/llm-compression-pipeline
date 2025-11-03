# 🎓 LLM Compression Pipeline: Knowledge Distillation for Question Answering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RVKarthikeyan/llm-compression-pipeline/blob/gjk/kd-colab.ipynb)

A complete pipeline for compressing large language models using knowledge distillation. This project extracts knowledge from PDF documents using a teacher model (Gemma-2-2B) and distills it into a smaller, faster student model (DistilBERT) for question-answering tasks.

## 🌟 Overview

This project demonstrates **knowledge distillation** - a technique to compress large language models while retaining most of their capabilities. The pipeline:

1. 📄 Extracts text from PDF documents
2. 🧠 Uses Gemma-2-2B (teacher) to generate high-quality QA pairs
3. 📚 Trains DistilBERT (student) on the generated dataset
4. ⚡ Produces a smaller, faster model specialized for your domain

**Key Benefits:**

- **40% smaller** model size compared to BERT
- **60% faster** inference time
- **Specialized knowledge** from your custom PDF documents
- **Low memory footprint** - runs on modest hardware

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Run all cells sequentially
3. Upload your PDF when prompted
4. Wait ~45-70 minutes for QA generation + 30-60 minutes for training

### Option 2: Local Installation

```bash
# Clone and setup
git clone https://github.com/RVKarthikeyan/llm-compression-pipeline.git
cd llm-compression-pipeline
pip install -r requirements.txt

# Run notebook
jupyter notebook kd-colab.ipynb
```

## 🔄 Pipeline Steps

1. **PDF Upload & Parsing**: Extract text from PDF documents using `pypdf`
2. **Text Chunking**: Split extracted text into **2000 token chunks** (optimized for Gemma's 8192 context window)
3. **Teacher Model Loading**: Load Gemma-2-2B-IT in **4-bit quantization** for memory efficiency
4. **QA Generation**: Generate question-answer pairs with reasoning in JSON format
5. **Data Storage**: Save ~94 QA pairs in `qa_dataset.json`
6. **Dataset Formatting**: Convert to HuggingFace dataset format for training
7. **Model Training**: Fine-tune DistilBERT on the generated dataset (3 epochs)
8. **Inference**: Test and deploy the trained extractive QA model

## 📋 Requirements

### Hardware

- **GPU**: NVIDIA GPU with 8GB+ VRAM (required for 4-bit quantization)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for models and datasets

### Software Dependencies

```bash
transformers>=4.36.0
datasets
accelerate
bitsandbytes
torch
pypdf
huggingface_hub
pandas
```

### Models Used

- **Teacher Model**: `google/gemma-2-2b-it` (2B parameters, 4-bit quantized)
- **Student Model**: `distilbert-base-uncased` (66M parameters)

## � Usage

### Basic Example

```python
from transformers import pipeline

# Load your trained model
qa_model = pipeline(
    "question-answering",
    model="./distilbert-qa-final",
    tokenizer="./distilbert-qa-final"
)

# Ask a question with context (REQUIRED!)
question = "What is a pointer in C?"
context = """A pointer is a variable that stores the memory address
of another variable. Pointers in C are declared using the asterisk
symbol and provide direct access to memory locations."""

# Get answer
result = qa_model(question=question, context=context)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.4f}")
```

**⚠️ Important**: Your model is an **extractive QA system** - you **must** always provide context text for it to extract answers from. It does NOT memorize answers from training.

### Step-by-Step Execution

1. **Run Colab or Jupyter Notebook**

   - Open `kd-colab.ipynb`
   - Execute cells sequentially

2. **Upload Your PDF**

   - Colab: Use the file upload widget
   - Local: Place PDF in project directory as `./book.pdf`

3. **Monitor Progress**

   - QA Generation: ~2-3 minutes per chunk (23 chunks for typical book)
   - Training: ~30-60 minutes (3 epochs)

4. **Access Your Model**
   - Trained model saved to `./distilbert-qa-final/`
   - QA dataset saved to `qa_dataset.json`

## 🏗️ Architecture

```
PDF Document
    ↓
Text Extraction (pypdf)
    ↓
Tokenization & Chunking (2000 tokens)
    ↓
Teacher Model (Gemma-2-2B-IT, 4-bit)
    ↓
QA Pair Generation (JSON format)
    ↓
Dataset Formatting (HuggingFace)
    ↓
Student Model Training (DistilBERT)
    ↓
Fine-tuned QA Model (66M params)
```

## 📊 Pipeline Details

### 1. PDF Parsing

```python
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
```

### 2. Text Chunking

- **Chunk Size**: **2000 tokens** (optimized for Gemma's 8192 context window)
- **Overlap**: 500 tokens (maintains context between chunks)
- **Why 2000?**: Leaves room for prompt template + 2000 token generation

### 3. Gemma Model Loading (4-bit Quantization)

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

**Memory Savings**: 4-bit quantization reduces model size by ~75% (from ~8GB to ~2GB)

### 4. QA Generation Format

Gemma generates 5-8 QA pairs per chunk in the following format:

```json
[
  {
    "question": "What is C's relationship with the UNIX operating system?",
    "answer": "C has been closely associated with the UNIX operating system...",
    "thinking_process": "The text explicitly states that C has been closely associated..."
  }
]
```

### 5. DistilBERT Training

- **Learning Rate**: 2e-5
- **Batch Size**: 8 per device
- **Epochs**: 3
- **Mixed Precision**: FP16 (if GPU available)
- **Evaluation**: Per epoch
- **Training Time**: ~30-60 minutes

## � Results

### Training Stats (Example: "The C Programming Language" PDF)

- **QA Pairs Generated**: 94 pairs from 23 chunks
- **Success Rate**: ~65-70% (some chunks had JSON parsing errors - this is normal)
- **Training Samples**: 84 pairs
- **Validation Samples**: 10 pairs
- **Final Model Size**: ~260MB

### Model Performance

| Model                 | Parameters | Size         | Inference Speed |
| --------------------- | ---------- | ------------ | --------------- |
| BERT-base             | 110M       | ~440MB       | 1.0x (baseline) |
| **DistilBERT (Ours)** | 66M        | ~260MB       | **1.6x faster** |
| Gemma-2-2B (Teacher)  | 2B         | ~2GB (4-bit) | 0.3x            |

### Sample Output

```
Question: What is DistilBERT?
Context: DistilBERT is a smaller, faster version of BERT...
Answer: "a smaller, faster version of BERT"
Confidence: 0.9847
```

## 📁 Project Structure

```
llm-compression-pipeline/
├── kd-colab.ipynb              # Main pipeline notebook
├── qa_dataset.json             # Generated QA pairs (94 pairs)
├── distilbert-qa-final/        # Trained model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
├── distilbert-qa-finetuned/    # Training checkpoints
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **Context Length Overflow**

```
Error: exceeded the model's predefined maximum length (8192)
```

**Solution**: The notebook already uses `CHUNK_SIZE = 2000` tokens. If you still see this, reduce it further to 1500.

#### 2. **Empty QA Dataset**

```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solution**: Re-run Step 10 (Process all chunks) and Step 11 (Save QA pairs).

#### 3. **GPU Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solution**:

- Use Google Colab with T4 GPU (free tier)
- Reduce batch size to 4
- Ensure 4-bit quantization is enabled

#### 4. **JSON Parsing Errors**

```
Warning: No JSON array found in response
```

**Solution**: This is **normal** - Gemma occasionally generates malformed JSON. The pipeline continues with valid chunks. ~30% failure rate is expected.

## 🎯 Key Features

### What Your Model Learned

Your model learned **HOW to extract answers from text**, not WHAT the answers are:

✅ **Skills Acquired:**

- Pattern recognition in technical text
- Question-answer relationships
- C programming vocabulary and concepts
- Text structure understanding

❌ **NOT Learned:**

- Memorizing the 94 QA pairs
- Generating answers without context
- Acting like ChatGPT

### Correct Usage

```python
# ✅ CORRECT: Always provide context
result = qa_model(
    question="What is a pointer?",
    context="A pointer is a variable that stores memory addresses..."
)

# ❌ WRONG: No context provided
result = qa_model(question="What is a pointer?")  # Will fail or give nonsense
```

## 💡 Tips & Best Practices

### For Better Results

1. **Provide Rich Context**: Give the model substantial text (100-500 words) to extract from
2. **Domain-Specific PDFs**: Train on technical documents for best specialized performance
3. **Question Clarity**: Ask specific, answerable questions
4. **Multiple PDFs**: Combine multiple documents for broader knowledge

### Customization Options

#### Adjust Chunk Size

```python
CHUNK_SIZE = 1500  # Reduce for smaller contexts
CHUNK_OVERLAP = 300  # Adjust overlap as needed
```

#### Modify Training Parameters

```python
training_args = TrainingArguments(
    num_train_epochs=5,  # More epochs for better learning
    per_device_train_batch_size=4,  # Reduce if OOM
    learning_rate=3e-5,  # Experiment with learning rate
)
```

#### Use Different Models

```python
TEACHER_MODEL = "google/gemma-7b-it"  # Larger teacher (needs more VRAM)
STUDENT_MODEL = "bert-base-uncased"  # Alternative student
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add retry logic for failed JSON parsing
- [ ] Support for DOCX, TXT formats
- [ ] Implement RAG (Retrieval-Augmented Generation)
- [ ] Add evaluation metrics (F1, EM scores)
- [ ] Batch processing for multiple PDFs
- [ ] Web interface for easy usage

## 📚 References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - Hugging Face's distillation approach
- [Gemma Models](https://blog.google/technology/developers/gemma-open-models/) - Google's open LLMs
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531) - Original KD paper by Hinton et al.
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - 4-bit quantization library

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **HuggingFace** for transformers library and model hosting
- **Google** for Gemma models
- **PyPDF** for PDF parsing
- **BitsAndBytes** for efficient quantization

## 📧 Contact

**Author**: RVKarthikeyan  
**GitHub**: [@RVKarthikeyan](https://github.com/RVKarthikeyan)  
**Repository**: [llm-compression-pipeline](https://github.com/RVKarthikeyan/llm-compression-pipeline)

---

⭐ **Star this repo** if you find it helpful!

**Note**: This is an educational project demonstrating knowledge distillation. The trained model's quality depends heavily on the input PDF and the number of QA pairs generated.

- Reduce number of chunks by increasing `CHUNK_SIZE`

## 📈 Performance Metrics

### Expected Results

- **QA Pairs per Chunk**: 5-8 pairs
- **Generation Time**: ~30-60 seconds per chunk (4-bit Gemma-2B)
- **Training Time**: ~30-60 minutes (depends on dataset size)
- **Model Size**: ~250MB (DistilBERT fine-tuned)

### Accuracy

- DistilBERT retains ~97% of BERT's performance
- Knowledge distillation from Gemma captures reasoning patterns
- Fine-tuning on domain-specific data improves accuracy

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📝 License

This project is for educational purposes. Please ensure you comply with the licenses of:

- Gemma Model (Google)
- DistilBERT (HuggingFace)
- Your source PDF materials

## 🙏 Acknowledgments

- Google for Gemma models
- HuggingFace for transformers library
- PyPDF2/pypdf for PDF parsing
- BitsAndBytes for quantization support

---

**Last Updated**: November 2025
