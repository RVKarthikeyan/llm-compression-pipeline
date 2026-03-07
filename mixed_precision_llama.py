"""
Mixed-Precision Quantization & ExecuTorch PTE Export for Llama models.

Uses ``optimum-cli export executorch`` to:
  1. Load a HuggingFace (or local) model.
  2. Apply 8da4w quantization to Linear layers.
  3. Apply 8w quantization to Embedding layers.
  4. Export to ExecuTorch with XNNPACK partitioning.
  5. Save ``.pte`` file + tokenizer to the output directory.

Prerequisites (must already be installed):
  torch (nightly CPU), torchao, executorch, optimum-executorch,
  transformers, sentencepiece, huggingface_hub.
"""

import os
import logging
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def run_mixed_precision_export(
    model_path: str,
    output_dir: str,
    hf_token: str | None = None,
    recipe: str = "xnnpack",
    qlinear: str = "8da4w",
    qembedding: str = "8w",
) -> dict:
    """Export a model to an ExecuTorch ``.pte`` file with mixed-precision quantization.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID (e.g. ``"meta-llama/Llama-3.2-1B-Instruct"``)
        or path to a local model directory.
    output_dir : str
        Directory where the ``.pte`` file and tokenizer will be saved.
    hf_token : str or None
        HuggingFace token.  Required for gated models.  Set in the
        subprocess environment as ``HF_TOKEN``.
    recipe : str
        ExecuTorch backend recipe (default ``"xnnpack"`` for ARM CPU).
    qlinear : str
        Quantization scheme for Linear layers
        (default ``"8da4w"`` -- 8-bit dynamic activation + 4-bit weight).
    qembedding : str
        Quantization scheme for Embedding layers
        (default ``"8w"`` -- 8-bit weight-only).

    Returns
    -------
    dict
        ``pte_path``   -- absolute path to the produced ``.pte`` file.
        ``output_dir`` -- the output directory (contains tokenizer files too).
        ``size_mb``    -- size of the ``.pte`` file in megabytes.

    Raises
    ------
    RuntimeError
        If the ``optimum-cli`` export process exits with a non-zero code.
    FileNotFoundError
        If no ``.pte`` file is found after a successful export.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f'optimum-cli export executorch'
        f' --model "{model_path}"'
        f' --task "text-generation"'
        f' --recipe "{recipe}"'
        f' --qlinear {qlinear}'
        f' --qembedding {qembedding}'
        f' --output_dir="{output_dir}"'
    )

    logger.info("Running mixed-precision export:\n  %s", cmd)

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    t0 = time.time()

    process = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        logger.info("[optimum-cli] %s", line.rstrip())

    returncode = process.wait()
    elapsed = time.time() - t0

    if returncode != 0:
        raise RuntimeError(
            f"optimum-cli export failed (exit code {returncode}) after {elapsed:.0f}s. "
            "Check logs above for details."
        )

    logger.info("Export succeeded in %.1f min", elapsed / 60)

    # ------------------------------------------------------------------
    # Locate the .pte file
    # ------------------------------------------------------------------
    pte_path = None
    for fname in os.listdir(output_dir):
        if fname.endswith(".pte"):
            pte_path = os.path.join(output_dir, fname)
            break

    if pte_path is None:
        raise FileNotFoundError(
            f"No .pte file found in {output_dir} after export."
        )

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    logger.info("PTE file: %s (%.1f MB)", pte_path, size_mb)

    # ------------------------------------------------------------------
    # Ensure tokenizer files are present (optimum-cli usually saves them,
    # but download separately as a fallback)
    # ------------------------------------------------------------------
    tokenizer_json = os.path.join(output_dir, "tokenizer.json")
    tokenizer_config = os.path.join(output_dir, "tokenizer_config.json")

    if not os.path.exists(tokenizer_json) or not os.path.exists(tokenizer_config):
        logger.info("Tokenizer files missing from output -- downloading separately.")
        from transformers import AutoTokenizer

        is_local = os.path.isdir(model_path)
        tok = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token if not is_local else None,
        )
        tok.save_pretrained(output_dir)

    return {
        "pte_path": os.path.abspath(pte_path),
        "output_dir": os.path.abspath(output_dir),
        "size_mb": size_mb,
    }
