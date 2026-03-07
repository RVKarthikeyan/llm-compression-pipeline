import os
import hashlib
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, BackgroundTasks
from huggingface_hub import HfApi, login, create_repo

from llmcompression_llama import run_knowledge_distillation
from llama_3_2_1b_instruct_prunned import run_pruning_pipeline

app = FastAPI(title="Llama Model Compression Pipeline API")

# Configuration
BASE_DIR = Path("workspace")
TARGET_REPO_NAME = "compressed_models"

BASE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store for authenticated users and job statuses
# In production, replace with a proper database
user_sessions = {}  # hashed_token -> { "hf_token": str, "username": str }
job_store = {}      # hashed_token -> { "job_id": str, "status": str, "message": str, "model_path": str | None }


def hash_token(hf_token: str) -> str:
    return hashlib.sha256(hf_token.encode()).hexdigest()


def get_user_dir(hashed_token: str) -> Path:
    user_dir = BASE_DIR / hashed_token
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


# ---------------------------------------------------------------------------
# Background pipeline runners
# ---------------------------------------------------------------------------

def _run_knowledge_distillation_pipeline(hf_token: str, hashed_token: str, pdf_path: str):
    """Background task: knowledge distillation (teacher 8B -> student 1B)."""
    user_dir = get_user_dir(hashed_token)
    job = job_store[hashed_token]

    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]

        job["status"] = "distilling"
        job["message"] = "Running knowledge distillation pipeline (teacher -> student)..."

        output_dir = str(user_dir / "kd_output")
        result = run_knowledge_distillation(
            hf_token=hf_token,
            pdf_path=pdf_path,
            output_dir=output_dir,
        )

        final_model_dir = result["final_model_dir"]

        # Upload to HuggingFace
        job["status"] = "uploading"
        job["message"] = "Uploading compressed model to Hugging Face..."
        repo_id = f"{username}/{TARGET_REPO_NAME}"
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            token=hf_token,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=final_model_dir,
            path_in_repo="kd-llama-domain-specialist",
            repo_id=repo_id,
            repo_type="model",
        )

        job["status"] = "completed"
        job["message"] = (
            f"Knowledge distillation complete. "
            f"Model uploaded to {repo_id}/kd-llama-domain-specialist "
            f"({result['num_training_examples']} training examples used)"
        )
        job["model_path"] = final_model_dir

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Pipeline failed: {str(e)}"


def _run_pruning_pipeline(hf_token: str, hashed_token: str):
    """Background task: LaCo pruning + LoRA healing."""
    user_dir = get_user_dir(hashed_token)
    job = job_store[hashed_token]

    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]

        job["status"] = "pruning"
        job["message"] = "Running pruning and LoRA healing pipeline..."

        output_dir = str(user_dir / "pruning_output")
        result = run_pruning_pipeline(
            hf_token=hf_token,
            output_dir=output_dir,
        )

        final_model_dir = result["final_model_dir"]

        # Upload to HuggingFace
        job["status"] = "uploading"
        job["message"] = "Uploading compressed model to Hugging Face..."
        repo_id = f"{username}/{TARGET_REPO_NAME}"
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            token=hf_token,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=final_model_dir,
            path_in_repo="pruned-llama-healed",
            repo_id=repo_id,
            repo_type="model",
        )

        job["status"] = "completed"
        job["message"] = (
            f"Pruning pipeline complete. "
            f"Model uploaded to {repo_id}/pruned-llama-healed "
            f"(layers: {result['num_layers_original']} -> {result['num_layers_pruned']})"
        )
        job["model_path"] = final_model_dir

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Pipeline failed: {str(e)}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/auth")
async def authenticate(hf_token: str = Form(...)):
    """Validate the HF token and create a workspace folder for the user."""
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        username = user_info["name"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Hugging Face token.")

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
    method: str = Form(..., description="Compression method: 'knowledge_distillation' or 'pruning'"),
    pdf_file: UploadFile = File(None),
):
    """Trigger a compression pipeline.

    Parameters
    ----------
    hf_token : str
        Hugging Face access token (must be authenticated via ``/auth`` first).
    method : str
        ``"knowledge_distillation"`` – Uses Llama 3.1 8B as teacher to distil
        domain knowledge from a PDF into Llama 3.2 1B.  Requires ``pdf_file``.

        ``"pruning"`` – Applies LaCo layer-collapse pruning to Llama 3.2 1B
        and heals via LoRA fine-tuning.
    pdf_file : UploadFile, optional
        PDF document (required when ``method="knowledge_distillation"``).
    """
    hashed = hash_token(hf_token)

    if hashed not in user_sessions:
        raise HTTPException(status_code=401, detail="Please authenticate first via /auth.")

    if method not in ("knowledge_distillation", "pruning"):
        raise HTTPException(
            status_code=400,
            detail="Invalid method. Choose 'knowledge_distillation' or 'pruning'.",
        )

    # Prevent re-triggering while a job is in progress
    active_statuses = ("queued", "distilling", "pruning", "uploading")
    if hashed in job_store and job_store[hashed]["status"] in active_statuses:
        raise HTTPException(
            status_code=409,
            detail=f"A pipeline is already running (status: {job_store[hashed]['status']}). Wait for it to finish.",
        )

    job_id = str(uuid.uuid4())
    job_store[hashed] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Pipeline queued.",
        "model_path": None,
    }

    if method == "knowledge_distillation":
        if pdf_file is None:
            raise HTTPException(
                status_code=400,
                detail="A PDF file is required for the knowledge_distillation method.",
            )
        # Save uploaded PDF to the user workspace
        user_dir = get_user_dir(hashed)
        pdf_path = str(user_dir / pdf_file.filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)

        background_tasks.add_task(
            _run_knowledge_distillation_pipeline, hf_token, hashed, pdf_path
        )
    else:
        background_tasks.add_task(_run_pruning_pipeline, hf_token, hashed)

    return {
        "status": "accepted",
        "job_id": job_id,
        "method": method,
        "message": f"{method} pipeline started in background.",
    }


@app.get("/status")
async def pipeline_status(hf_token: str):
    """Check pipeline progress / whether the model has been generated."""
    hashed = hash_token(hf_token)

    if hashed not in user_sessions:
        raise HTTPException(status_code=401, detail="Please authenticate first via /auth.")

    if hashed not in job_store:
        return {"status": "no_job", "message": "No pipeline has been triggered yet."}

    job = job_store[hashed]
    model_ready = job["model_path"] is not None and Path(job["model_path"]).exists()

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "message": job["message"],
        "model_ready": model_ready,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
