import os
import hashlib
import shutil
import uuid
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from huggingface_hub import HfApi, login, snapshot_download, create_repo
from pathlib import Path

from pruning_gemma3_1b_it import run_pruning_pipeline
from mixed_precision_quantization_executorch import run_mixed_precision_quantization

app = FastAPI(title="Gemma Model Compression Pipeline API")

# Configuration
BASE_DIR = Path("workspace")
TARGET_REPO_NAME = "pte_models"

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


def run_pipeline(hf_token: str, hashed_token: str, model_id: str):
    """
    Orchestrates: prune + LoRA heal -> quantize to PTE -> create repo -> upload.
    """
    user_dir = get_user_dir(hashed_token)
    job = job_store[hashed_token]

    try:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]

        # 1. Pruning + LoRA healing
        job["status"] = "pruning"
        job["message"] = "Running pruning and LoRA healing..."

        pruned_output_dir = str(user_dir / "compressed_gemma")
        lora_output_dir = str(user_dir / "lora_adapters")
        final_merged_dir = str(user_dir / "final_healed_model")
        zip_output = str(user_dir / "compressed_gemma_final.zip")

        pruning_result = run_pruning_pipeline(
            model_path=model_id,
            hf_token=hf_token,
            output_dir=pruned_output_dir,
            lora_output_dir=lora_output_dir,
            final_merged_dir=final_merged_dir,
            zip_output=zip_output,
        )

        # 2. Mixed-precision quantization + ExecuTorch PTE conversion
        job["status"] = "quantizing"
        job["message"] = "Running mixed-precision quantization and PTE conversion..."

        pte_output_dir = str(user_dir / "pte_output")
        model_short_name = model_id.split("/")[-1]
        pte_filename = f"{model_short_name}_int8.pte"

        quantization_result = run_mixed_precision_quantization(
            model_path=pruning_result["final_model_dir"],
            hf_token=hf_token,
            output_dir=pte_output_dir,
            output_filename=pte_filename,
        )

        pte_output_path = quantization_result["output_path"]

        # 3. Create private HF repo and upload
        job["status"] = "uploading"
        job["message"] = "Uploading PTE file to Hugging Face..."
        repo_id = f"{username}/{TARGET_REPO_NAME}"
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            token=hf_token,
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=pte_output_path,
            path_in_repo=pte_filename,
            repo_id=repo_id,
            repo_type="model",
        )

        job["status"] = "completed"
        job["message"] = (
            f"Pipeline completed. PTE uploaded to {repo_id}/{pte_filename} "
            f"({quantization_result['size_mb']:.1f} MB)"
        )
        job["pte_path"] = pte_output_path

    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Pipeline failed: {str(e)}"


@app.post("/auth")
async def authenticate(hf_token: str = Form(...)):
    """
    Validate the HF token, create a workspace folder for the user.
    """
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
    model_id: str = Form(...),
):
    """
    Trigger the full compression pipeline: prune -> quantize -> upload PTE.
    """
    hashed = hash_token(hf_token)

    if hashed not in user_sessions: 
        raise HTTPException(status_code=401, detail="Please authenticate first via /auth.")

    # Prevent re-triggering if a job is already running
    if hashed in job_store and job_store[hashed]["status"] in (
        "downloading", "pruning", "quantizing", "uploading"
    ):
        raise HTTPException(
            status_code=409,
            detail=f"A pipeline is already running (status: {job_store[hashed]['status']}). Wait for it to finish.",
        )

    job_id = str(uuid.uuid4())
    job_store[hashed] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Pipeline queued.",
        "pte_path": None,
    }

    background_tasks.add_task(run_pipeline, hf_token, hashed, model_id)

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Pipeline started in background.",
    }


@app.get("/status")
async def pipeline_status(hf_token: str):
    """
    Check whether the PTE file has been generated / pipeline progress.
    """
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


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
