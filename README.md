# LLM Compression Pipeline

A capstone-grade, end-to-end monorepo for building domain-specialized compact language models and deploying them to mobile devices through a structured compression workflow.

The repository contains:

- mobile-app: Flutter client for authentication, orchestration, monitoring, retrieval, local indexing, and on-device inference integration.
- api-backend: FastAPI service that executes pruning, distillation, quantization-export, and artifact publication.

## Table of Contents

1. Project Overview
2. Problem and Design Goals
3. Repository Structure
4. Full System Architecture
5. End-to-End Workflow
6. Theory: Core Model Compression Methods
7. Theory: Retrieval and Data Preparation Methods
8. Theory: Mobile Inference and Runtime Integration
9. Technology Stack and Module-Level Mapping
10. API Reference
11. How to Use This Repository
12. How to Use the Mobile App
13. Reproducibility and Experiment Tracking
14. Validation and Evaluation Plan
15. Operational Risks and Troubleshooting
16. Security and Data Handling Notes
17. Academic Project Notice
18. Acknowledgments

## 1. Project Overview

This project combines model compression research methods with practical software engineering to deliver deployable LLM artifacts for constrained environments.

The backend executes a staged compression pipeline:

1. Structured pruning with healing.
2. Domain-oriented knowledge distillation.
3. Mixed-precision quantization and ExecuTorch export.
4. Artifact upload to a private Hugging Face model repository.

The Flutter application provides:

- Credential input and secure storage.
- Pipeline trigger and status polling.
- PDF ingestion and local chunk store.
- Model download and native inference bridge.

## 2. Problem and Design Goals

### 2.1 Core Problem

Large language models are expensive in memory, latency, and compute. Direct deployment on edge/mobile platforms is often infeasible without aggressive optimization. Domain adaptation also requires robust pipelines that are reproducible and operable by non-terminal users.

### 2.2 Design Goals

- Preserve useful task quality while reducing deploy-time footprint.
- Transfer teacher behavior into a smaller student model.
- Keep adaptation cost manageable through parameter-efficient training.
- Package outputs in mobile-compatible format.
- Provide a practical UI-driven workflow for a capstone demonstration.

### 2.3 Non-Goals

- Full distributed production MLOps platform.
- Training a foundation model from scratch.
- Multi-tenant enterprise-grade identity and access management.

## 3. Repository Structure

llm-compression-pipeline/
- api-backend/
  - api.py
  - pruning_pipeline.py
  - knowledge_distillation_pipeline.py
  - quantization_export_pipeline.py
  - requirements.txt
  - setup.sh
- mobile-app/
  - lib/
    - providers/
    - services/
    - views/
    - models/
    - main.dart
  - android/
  - ios/
  - web/
  - windows/
  - linux/
  - macos/
- README.md
- .gitignore

## 4. Full System Architecture

### 4.1 Logical Components

- Client Layer: Flutter app with Riverpod-managed state and platform channels.
- API Layer: FastAPI service with background job orchestration.
- ML Layer: Modular Python pipelines for pruning, distillation, export.
- Storage Layer: User-scoped workspace folders plus Hugging Face artifact hosting.

### 4.2 Data Flow

1. User authenticates from mobile app.
2. PDF is selected and uploaded to backend.
3. Backend executes compression stages asynchronously.
4. Job status is polled from app.
5. Final .pte artifact is uploaded and can be downloaded by app.
6. App can load model via native ExecuTorch bridge.

## 5. End-to-End Workflow

### Stage A: Authentication and session bootstrap

- Endpoint: POST /auth
- Backend verifies Hugging Face token.
- W and B environment variables are set.
- User workspace is created under api-backend/workspace/<hash>/.

### Stage B: Pipeline trigger

- Endpoint: POST /trigger-pipeline
- PDF is saved to workspace.
- Background task is queued.

### Stage C: Compression execution pipeline

1. Pruning stage from pruning_pipeline.py.
2. Distillation stage from knowledge_distillation_pipeline.py.
3. Quantization and export stage from quantization_export_pipeline.py.
4. Upload stage to private Hugging Face model repository.

### Stage D: Status and retrieval

- Endpoint: GET /status returns stage and progress message.
- App downloads .pte from Hugging Face via authenticated requests.

### Stage E: Optional embedding service

- Endpoint: POST /embed
- PDF text extraction and chunking.
- all-MiniLM-L6-v2 embeddings generated for downstream retrieval storage.

## 6. Theory: Core Model Compression Methods

### 6.1 Structured Pruning

Structured pruning removes model components at architecture level, such as complete layers, which is deployment-friendly relative to unstructured sparsity.

In this implementation:

- Adjacent transformer layers are merged by parameter averaging.
- layer_idx fields are corrected after merge to keep cache indexing consistent.

Why this matters:

- Dense graph is preserved.
- Inference kernels remain conventional.
- Memory and compute can drop significantly at deployment.

### 6.2 Pruning Shock and Healing

After structural edits, model behavior can degrade due to representational mismatch. The project addresses this with a LoRA-based healing phase that recovers capability without full fine-tuning cost.

### 6.3 Knowledge Distillation (Black-Box Practical Mode)

Knowledge distillation transfers teacher behavior to student behavior.

In this project, distillation follows a practical black-box style:

- Teacher internals are not required at training time.
- Teacher-generated outputs over synthetic prompts become supervision targets.
- Student learns from response behavior grounded in document-derived context.

### 6.4 LoRA (Low-Rank Adaptation)

LoRA introduces trainable low-rank adapter matrices into selected projection layers while freezing most base weights.

Advantages:

- Lower VRAM and training cost.
- Faster adaptation iterations.
- Easy adapter merge for final deployable checkpoints.

### 6.5 Mixed-Precision Quantization and Export

Mixed precision balances accuracy and efficiency by assigning different precision strategies to different operator categories.

In this implementation:

- Linear layers use 8da4w style configuration.
- Embedding layers use 8w style configuration.
- Export path targets ExecuTorch and mobile-ready .pte artifacts.

### 6.6 Why these methods are combined

Each method optimizes a different bottleneck:

- Structured pruning: architecture-level reduction.
- Distillation: capability transfer.
- LoRA: efficient adaptation.
- Quantization: deploy-time speed and memory efficiency.

The combined effect is stronger than any single method alone for edge deployment.

## 7. Theory: Retrieval and Data Preparation Methods

### 7.1 PDF text extraction and chunking

The backend uses PyMuPDF for extraction. The app also supports local PDF extraction via Syncfusion for local chunking workflows.

### 7.2 Chunking strategy

ObjectBox service implements multiple fallback chunking strategies:

- Section-aware splitting.
- Paragraph-aware chunking with overlap.
- Sentence-group fallback.

This improves retrieval robustness when document structure varies.

### 7.3 Local retrieval heuristics in mobile app

ObjectBox-backed query logic includes:

- Stopword filtering.
- N-gram keyword generation.
- Domain synonym expansion.
- Medical term mapping and reverse lookup support.

This supports practical context retrieval before inference calls.

### 7.4 Embedding theory reference

Sentence-transformers all-MiniLM-L6-v2 generates normalized dense vectors (384 dimensions), enabling semantic similarity search patterns and vector index integration.

## 8. Theory: Mobile Inference and Runtime Integration

### 8.1 ExecuTorch runtime integration

Android native integration is implemented in MainActivity.kt with LlmModule.

The native layer handles:

- Model and tokenizer load.
- Token streaming callbacks.
- Cache reset and stop controls.
- Chat-template-aware prompt formatting support.

### 8.2 Platform channel bridge

Flutter uses MethodChannel to communicate with native inference methods.

Why this architecture is used:

- Leverages Flutter for cross-platform UI.
- Keeps heavy inference runtime in native layer.
- Enables token-level streaming updates to UI.

### 8.3 State management and UI consistency

Riverpod notifiers manage model state, chat state, settings state, and async pipeline status transitions. This supports deterministic UI behavior and clean state transitions.

## 9. Technology Stack and Module-Level Mapping

## 9.1 Backend libraries and purpose

- fastapi, uvicorn: API and ASGI runtime.
- transformers, torch, bitsandbytes, accelerate: model loading and optimization.
- peft, trl, datasets: LoRA and supervised fine-tuning flows.
- PyMuPDF: document extraction.
- sentence-transformers: embedding generation.
- huggingface_hub: auth and artifact publication.
- wandb: experiment logging and run tracking.
- python-multipart: file upload support.

## 9.2 Flutter libraries and purpose

- flutter_riverpod: state and dependency management.
- dio, http: network client and file download.
- flutter_secure_storage: sensitive credential storage.
- objectbox, objectbox_flutter_libs: local persistent store and retrieval cache.
- file_picker: PDF selection.
- syncfusion_flutter_pdf: PDF text extraction in app.
- pointycastle: AES encryption utilities.
- path, path_provider: file paths and local storage locations.

## 9.3 Repository module mapping

- api-backend/api.py:
  - Session handling, job orchestration, endpoints, upload.
- api-backend/pruning_pipeline.py:
  - Structured pruning and healing flow.
- api-backend/knowledge_distillation_pipeline.py:
  - Teacher-student distillation and student adaptation.
- api-backend/quantization_export_pipeline.py:
  - Quantization plus ExecuTorch export.
- mobile-app/lib/providers/app_providers.dart:
  - Riverpod providers, model/chat state, secure settings.
- mobile-app/lib/services/objectbox_service.dart:
  - Chunk persistence and retrieval-oriented preprocessing.
- mobile-app/lib/services/backend_service.dart:
  - API requests and model artifact download.
- mobile-app/android/app/src/main/kotlin/com/example/my_ai/MainActivity.kt:
  - Native ExecuTorch bridge and streaming inference.

## 10. API Reference

### POST /auth

Form fields:

- hf_token
- wandb_api_key

Returns:

- status
- username
- user_hash

### POST /trigger-pipeline

Form fields:

- hf_token
- pdf_file

Returns:

- status
- job_id
- message

### GET /status

Query:

- hf_token

Returns:

- job_id
- status
- message
- pte_ready

### POST /embed

Form fields:

- hf_token
- pdf_file
- chunk_size optional
- chunk_overlap optional

Returns:

- model
- dimensions
- count
- chunks with embeddings

### GET /health

Returns:

- status

## 11. How to Use This Repository

### 11.1 Backend setup (manual, cross-platform)

From repository root:

1. cd api-backend
2. python -m venv .venv
3. Activate environment
   - Windows PowerShell: .\.venv\Scripts\Activate.ps1
   - Linux/Mac: source .venv/bin/activate
4. pip install --upgrade pip
5. pip install -r requirements.txt
6. uvicorn api:app --host 0.0.0.0 --port 8000 --reload

### 11.2 Backend setup (script-based)

1. cd api-backend
2. chmod +x setup.sh
3. ./setup.sh

Notes:

- setup.sh assumes bash and Linux utilities.
- For Windows native, use manual setup or WSL.

### 11.3 Mobile app setup

1. cd mobile-app
2. flutter clean
3. flutter pub get
4. flutter run

### 11.4 Developer workflow recommendation

- Keep backend terminal running.
- Use emulator/device with reachable backend URL.
- Verify /health endpoint before app-side auth.

## 12. How to Use the Mobile App

### 12.1 Prerequisites

- Running backend server.
- Hugging Face access token.
- Weights and Biases API key.
- Domain PDF file.

### 12.2 In-app sequence

1. Open Train and Download flow.
2. Enter HF token, W and B key, backend URL.
3. Authenticate.
4. Pick PDF document.
5. Trigger pipeline.
6. Monitor status until completed.
7. Download generated .pte model.
8. Load model into native runtime from app controls.

### 12.3 Backend URL notes

- Android emulator default: http://10.0.2.2:8000
- Physical device: use host LAN IP and open firewall port.

## 13. Reproducibility and Experiment Tracking

- Seed control is implemented in distillation pipeline.
- W and B run metadata is logged for major training phases.
- Recommended practice:
  - Pin model revisions.
  - Persist config snapshots per job_id.
  - Archive stage outputs for auditability.

## 14. Validation and Evaluation Plan

Suggested evaluation dimensions:

- Quality:
  - Domain QA correctness and consistency.
  - Hallucination rate.
  - Response completeness.
- Efficiency:
  - Stage runtime distribution.
  - GPU memory profile.
  - Final artifact size.
  - On-device latency and memory use.
- Robustness:
  - Failure recovery behavior.
  - Sensitivity to prompt distribution shift.

## 15. Operational Risks and Troubleshooting

### Common failure classes

- Authentication failure for gated models.
- OOM during pruning or distillation.
- Quantization/export dependency mismatch.
- Missing tokenizer artifacts in export output.
- App-to-backend network misconfiguration.

### Practical remediation

- Validate token permissions and model access.
- Reduce training size or adjust batch/accumulation.
- Verify optimum-cli resolution in export venv.
- Confirm backend /health before pipeline trigger.
- Use verbose logs for per-stage diagnostics.

## 16. Security and Data Handling Notes

- Treat HF tokens and W and B keys as secrets.
- Use secure storage on client side.
- Avoid plaintext logging of credentials.
- Prefer private artifact repositories.
- Define retention policy for workspace artifacts.

## 17. Academic Project Notice

This repository is part of a college capstone project. It is shared for academic learning, demonstration, and project evaluation.

No formal open-source license is declared at this stage.

If you intend to reuse substantial portions beyond academic context, consult the project team first and ensure compliance with upstream model and dataset terms.

## 18. Acknowledgments

Project team:

- [Jaswin Kumar N R](https://github.com/gjaswin)
- [Karthikeyan R V](https://github.com/RVKarthikeyan)
- [Navamohan M](https://github.com/NavamohanM)
- [Mohammed Thalha M](https://github.com/devmohammedthalha)

The team acknowledges the open ML ecosystem, including Hugging Face, PyTorch, ExecuTorch, Flutter, and the research community that enabled these methods.
