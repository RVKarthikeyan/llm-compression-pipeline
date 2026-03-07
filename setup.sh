#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh  –  First-time setup for the Llama compression pipeline
#
# Creates two virtual environments:
#   .venv_main  –  API server, pruning, knowledge distillation
#                  (CUDA torch + HuggingFace ML stack)
#   .venv_pte   –  ExecuTorch mixed-precision PTE conversion
#                  (nightly CPU torch + executorch + optimum-executorch)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_MAIN="$SCRIPT_DIR/.venv_main"
VENV_PTE="$SCRIPT_DIR/.venv_pte"
NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cpu"

# ── Helpers ────────────────────────────────────────────────────────────────
info()  { printf "\n\033[1;34m=> %s\033[0m\n" "$*"; }
ok()    { printf "\033[1;32m   ✓ %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m   ! %s\033[0m\n" "$*"; }

# ── System dependencies ───────────────────────────────────────────────────
info "Checking system dependencies"

missing=()
command -v tesseract >/dev/null 2>&1 || missing+=(tesseract-ocr)
command -v pdftoppm  >/dev/null 2>&1 || missing+=(poppler-utils)
command -v python3   >/dev/null 2>&1 || missing+=(python3)

if [ ${#missing[@]} -gt 0 ]; then
    warn "Missing system packages: ${missing[*]}"
    echo "   Install them with:"
    echo "     sudo apt-get update && sudo apt-get install -y ${missing[*]}"
    echo ""
    read -rp "   Attempt to install now? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        sudo apt-get update && sudo apt-get install -y "${missing[@]}"
        ok "System packages installed"
    else
        warn "Skipping – some pipeline steps may fail without these packages"
    fi
else
    ok "tesseract, poppler-utils, python3 found"
fi

# ── venv_main: API + Pruning + Knowledge Distillation ─────────────────────
info "Creating main venv at $VENV_MAIN"

python3 -m venv "$VENV_MAIN"
"$VENV_MAIN/bin/pip" install --upgrade pip setuptools wheel -q

info "Installing main requirements"
"$VENV_MAIN/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

ok "Main venv ready: $VENV_MAIN"

# ── venv_pte: ExecuTorch PTE Conversion ───────────────────────────────────
info "Creating PTE venv at $VENV_PTE"

python3 -m venv "$VENV_PTE"
"$VENV_PTE/bin/pip" install --upgrade pip setuptools wheel -q

# Step 1: Pre-install executorch's PyPI dependencies
#         (the nightly index doesn't host these, causing resolver failures)
info "Installing executorch PyPI dependencies"
"$VENV_PTE/bin/pip" install --no-cache-dir \
    "coremltools==9.0" \
    "mpmath==1.3.0" \
    expecttest flatbuffers hypothesis kgb \
    parameterized "pytest<9.0" pytest-xdist "pytest-rerunfailures==15.1" \
    pytest-json-report pytorch-tokenizers ruamel.yaml tabulate \
    hydra-core omegaconf pandas "scikit-learn>=1.5"

# Step 2: torch nightly (CPU – no CUDA needed for export)
info "Installing torch nightly (CPU)"
"$VENV_PTE/bin/pip" install --no-cache-dir --pre torch --index-url "$NIGHTLY_INDEX"

# Step 3: torchao nightly (must match torch version)
info "Installing torchao nightly"
"$VENV_PTE/bin/pip" install --no-cache-dir --pre torchao --index-url "$NIGHTLY_INDEX"

# Step 4: executorch nightly (--no-deps to skip nightly-index-only resolution)
info "Installing executorch nightly"
"$VENV_PTE/bin/pip" install --no-cache-dir --no-deps --pre executorch --index-url "$NIGHTLY_INDEX"

# Step 5: optimum-executorch (--no-deps – its version pins are too strict)
info "Installing optimum-executorch"
"$VENV_PTE/bin/pip" install --no-cache-dir --no-deps optimum-executorch

# Step 6: HuggingFace stack (accelerate with --no-deps to avoid pulling CUDA torch)
info "Installing HuggingFace stack"
"$VENV_PTE/bin/pip" install -q -U transformers tokenizers sentencepiece huggingface_hub optimum
"$VENV_PTE/bin/pip" install -q --no-deps accelerate

ok "PTE venv ready: $VENV_PTE"

# ── Verification ──────────────────────────────────────────────────────────
info "Verifying main venv"
"$VENV_MAIN/bin/python" -c "
import torch, transformers, peft, trl, datasets, bitsandbytes
import fastapi, pytesseract
print(f'  torch         : {torch.__version__}')
print(f'  transformers  : {transformers.__version__}')
print(f'  peft          : {peft.__version__}')
print(f'  trl           : {trl.__version__}')
print(f'  bitsandbytes  : {bitsandbytes.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"
ok "Main venv verified"

info "Verifying PTE venv"
"$VENV_PTE/bin/python" -c "
import torch
print(f'  torch        : {torch.__version__}')
assert 'cpu' in torch.__version__ or 'dev' in torch.__version__, \
    f'WRONG TORCH BUILD: {torch.__version__} (expected nightly CPU)'
try:
    import torchao; print(f'  torchao      : {torchao.__version__}')
except Exception as e: print(f'  torchao      : ERROR ({e})')
try:
    import executorch; print(f'  executorch   : {getattr(executorch, \"__version__\", \"installed\")}')
except Exception as e: print(f'  executorch   : ERROR ({e})')
try:
    import optimum.executorch; print(f'  optimum-et   : installed')
except Exception as e: print(f'  optimum-et   : ERROR ({e})')
import shutil
cli = shutil.which('optimum-cli')
print(f'  optimum-cli  : {cli if cli else \"NOT FOUND (expected)\"}')
" 2>&1 || warn "PTE verification had issues – check output above"

# Check that venv_pte has its own optimum-cli
if [ -x "$VENV_PTE/bin/optimum-cli" ]; then
    ok "optimum-cli found at $VENV_PTE/bin/optimum-cli"
else
    warn "optimum-cli not found in PTE venv – PTE export may fail"
fi

ok "PTE venv verified"

# ── Summary ───────────────────────────────────────────────────────────────
info "Setup complete"
echo ""
echo "  Main venv (API + pruning + KD):"
echo "    source $VENV_MAIN/bin/activate"
echo ""
echo "  Start the API server:"
echo "    $VENV_MAIN/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000"
echo ""
echo "  The API will automatically use $VENV_PTE for PTE conversion."
echo ""
