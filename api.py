import os
import hashlib
import shutil
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, BackgroundTasks
from huggingface_hub import HfApi, login, create_repo
from sentence_transformers import SentenceTransformer

from llama_3_2_1b_instruct_prunned import run_pruning_pipeline
from llmcompression_llama import run_knowledge_distillation, _extract_pdf_text, _chunk_text
from mixed_precision_llama import run_mixed_precision_export

logger = logging.getLogger(__name__)

app = FastAPI(title="Llama Model Compression Pipeline API")

# Load embedding model at startup (384-dim, ~80 MB)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configuration
BASE_DIR = Path("workspace")
TARGET_REPO_NAME = "compressed_models"
PTE_VENV = Path(__file__).resolve().parent / ".venv_pte"  # created by setup.sh

BASE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store for authenticated users and job statuses
# In production, replace with a proper database
user_sessions = {}  # hashed_token -> { "hf_token": str, "username": str }
job_store = {}      # hashed_token -> { "job_id": str, "status": str, "message": str, "pte_path": str | None }


def hash_token(hf_token: str) -> str:
    return hashlib.sha256(hf_token.encode()).hexdigest()


def get_user_dir(hashed_token: str) -> Path:
    user_dir = BASE_DIR / hashed_token
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


# ---------------------------------------------------------------------------
# Background pipeline: pruning -> KD -> mixed-precision PTE -> upload
# ---------------------------------------------------------------------------

def run_pipeline(hf_token: str, hashed_token: str, pdf_path: str, job_id: str):
    """Orchestrate the full compression pipeline.

    Steps
    -----
    1. **Pruning** – LaCo layer collapse + LoRA healing on Llama 3.2 1B.
    2. **Knowledge Distillation** – Teacher (8B) generates domain-specific
       training data from the PDF, then fine-tunes the *pruned* student.
    3. **Mixed-Precision PTE Export** – 8da4w + 8w quantisation via
       ExecuTorch / XNNPACK.
    4. **Upload** – Push the contents of the pte_output directory to a
       private HuggingFace repo under a folder named by job id.
    """
    user_dir = get_user_dir(hashed_token)
    job = job_store[hashed_token]

    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]

        # ------------------------------------------------------------------
        # 1. Pruning
        # ------------------------------------------------------------------
        job["status"] = "pruning"
        job["message"] = "Step 1/4: Pruning model (LaCo + LoRA healing)..."

        pruning_output_dir = str(user_dir / "pruning_output")
        pruning_result = run_pruning_pipeline(
            hf_token=hf_token,
            output_dir=pruning_output_dir,
        )

        pruned_model_dir = pruning_result["final_model_dir"]

        # ------------------------------------------------------------------
        # 2. Knowledge Distillation (student = pruned model)
        # ------------------------------------------------------------------
        job["status"] = "distilling"
        job["message"] = "Step 2/4: Knowledge distillation (teacher -> pruned student)..."

        kd_output_dir = str(user_dir / "kd_output")
        kd_result = run_knowledge_distillation(
            hf_token=hf_token,
            pdf_path=pdf_path,
            output_dir=kd_output_dir,
            student_model_name=pruned_model_dir,
        )

        kd_model_dir = kd_result["final_model_dir"]

        # ------------------------------------------------------------------
        # 3. Mixed-Precision PTE Export
        # ------------------------------------------------------------------
        job["status"] = "quantizing"
        job["message"] = "Step 3/4: Mixed-precision quantization + PTE export..."

        pte_output_dir = str(user_dir / "pte_output")
        pte_result = run_mixed_precision_export(
            model_path=kd_model_dir,
            output_dir=pte_output_dir,
            hf_token=hf_token,
            pte_venv=str(PTE_VENV),
        )

        pte_path = pte_result["pte_path"]

        # ------------------------------------------------------------------
        # 4. Upload to HuggingFace
        # ------------------------------------------------------------------
        job["status"] = "uploading"
        job["message"] = "Step 4/4: Uploading PTE to Hugging Face..."

        repo_id = f"{username}/{TARGET_REPO_NAME}"
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            token=hf_token,
            exist_ok=True,
        )

        api.upload_folder(
            folder_path=pte_output_dir,
            path_in_repo=job_id,
            repo_id=repo_id,
            repo_type="model",
        )

        job["status"] = "completed"
        job["message"] = (
            f"Pipeline complete. PTE output uploaded to {repo_id}/{job_id}/ "
            f"({pte_result['size_mb']:.1f} MB). "
            f"Layers: {pruning_result['num_layers_original']} -> "
            f"{pruning_result['num_layers_pruned']}. "
            f"KD examples: {kd_result['num_training_examples']}."
        )
        job["pte_path"] = pte_path

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Pipeline failed: {str(e)}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/auth")
async def authenticate(hf_token: str = Form(...), wandb_api_key: str = Form(...)):
    """Validate the HF token, set the W&B API key, and create a workspace folder."""
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        username = user_info["name"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Hugging Face token.")

    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_ENTITY"] = "gjk-pipeline"

    hashed = hash_token(hf_token)
    get_user_dir(hashed)

    user_sessions[hashed] = {"hf_token": hf_token, "username": username}

    return {
        "status": "authenticated",
        "username": username,
        "user_hash": hashed,
    }


@app.post("/trigger-pipeline")
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
    hf_token: str = Form(...),
    pdf_file: UploadFile = File(...),
):
    """Trigger the full compression pipeline.

    The pipeline runs four steps sequentially in the background:

    1. **Pruning** -- LaCo layer-collapse + LoRA healing on Llama 3.2 1B.
    2. **Knowledge Distillation** -- Llama 3.1 8B teacher generates
       domain-specific Q&A from the uploaded PDF and fine-tunes the pruned
       student.
    3. **Mixed-Precision Export** -- 8da4w / 8w quantisation → ExecuTorch
       ``.pte`` (XNNPACK backend).
    4. **Upload** -- Push ``.pte`` to a private HuggingFace repo.

    Parameters
    ----------
    hf_token : str
        HuggingFace access token (must be authenticated via ``/auth`` first).
    pdf_file : UploadFile
        Domain-specific PDF document used for the KD training-data generation.
    """
    hashed = hash_token(hf_token)

    if hashed not in user_sessions:
        raise HTTPException(status_code=401, detail="Please authenticate first via /auth.")

    # Prevent re-triggering while a job is in progress
    active_statuses = ("queued", "pruning", "distilling", "quantizing", "uploading")
    if hashed in job_store and job_store[hashed]["status"] in active_statuses:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A pipeline is already running (status: {job_store[hashed]['status']}). "
                "Wait for it to finish."
            ),
        )

    # Save uploaded PDF to user workspace
    user_dir = get_user_dir(hashed)
    pdf_path = str(user_dir / pdf_file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)

    job_id = str(uuid.uuid4())
    job_store[hashed] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Pipeline queued.",
        "pte_path": None,
    }

    background_tasks.add_task(run_pipeline, hf_token, hashed, pdf_path, job_id)

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Pipeline started in background (pruning -> KD -> PTE export -> upload).",
    }


@app.get("/status")
async def pipeline_status(hf_token: str):
    """Check pipeline progress / whether the PTE file has been generated."""
    hashed = hash_token(hf_token)

    if hashed not in user_sessions:
        raise HTTPException(status_code=401, detail="Please authenticate first via /auth.")

    if hashed not in job_store:
        return {"status": "no_job", "message": "No pipeline has been triggered yet."}

    job = job_store[hashed]
    pte_ready = job["pte_path"] is not None and Path(job["pte_path"]).exists()

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "message": job["message"],
        "pte_ready": pte_ready,
    }


@app.post("/embed")
async def generate_embeddings(
    hf_token: str = Form(...),
    pdf_file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
):
    """Extract text from a PDF, chunk it, and return embeddings for each chunk.

    The returned embeddings are float32 vectors (384 dimensions) suitable for
    storage in ObjectBox with @HnswIndex(dimensions: 384).

    Parameters
    ----------
    hf_token : str
        HuggingFace token (must be authenticated via /auth first).
    pdf_file : UploadFile
        PDF document to embed.
    chunk_size : int
        Character length per chunk (default 500).
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks (default 50).
    """
    hashed = hash_token(hf_token)

    if hashed not in user_sessions:
        raise HTTPException(status_code=401, detail="Please authenticate first via /auth.")

    # Save uploaded PDF
    user_dir = get_user_dir(hashed)
    pdf_path = str(user_dir / pdf_file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)

    # Extract text from PDF
    try:
        full_text = _extract_pdf_text(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text from PDF: {e}")
    finally:
        os.remove(pdf_path)

    if not full_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the PDF.")

    # Chunk with overlap
    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    for i in range(0, len(full_text), step):
        chunk = full_text[i : i + chunk_size]
        if len(chunk) > chunk_size // 4:
            chunks.append(chunk)

    if not chunks:
        raise HTTPException(status_code=422, detail="PDF text too short to produce any chunks.")

    # Generate embeddings
    embeddings = embedding_model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

    return {
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "count": len(chunks),
        "chunks": [
            {
                "index": i,
                "text": chunk,
                "embedding": emb.tolist(),
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ],
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
