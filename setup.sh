#!/usr/bin/env bash
# Bootstrap the mosquito wingbeat research environment on Apple Silicon (M1 Pro).
# Idempotent: safe to re-run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PY=${PYTHON:-python3}
VENV_DIR=".venv"

echo "==> Project root: $PROJECT_ROOT"
echo "==> Python:       $($PY --version)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "WARN: this script is tuned for macOS / Apple Silicon. Continuing anyway."
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "WARN: this machine is not arm64. MPS will not be available."
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating venv at $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
else
  echo "==> Reusing existing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip wheel setuptools >/dev/null

echo "==> Installing requirements (this can take a few minutes)"
pip install -r requirements.txt

echo "==> Verifying PyTorch + MPS"
python - <<'PY'
import torch
print(f"  torch         : {torch.__version__}")
print(f"  mps available : {torch.backends.mps.is_available()}")
print(f"  mps built     : {torch.backends.mps.is_built()}")
if torch.backends.mps.is_available():
    x = torch.randn(8, 8, device="mps")
    y = (x @ x.T).sum().item()
    print(f"  mps smoke-test: ok (matmul sum={y:.3f})")
else:
    print("  mps smoke-test: SKIPPED — falling back to CPU")
PY

cat <<'EOF'

==> Done.

Next steps:
  1) Activate the venv in your current shell:
       source .venv/bin/activate

  2) Set the MPS fallback env var (handles ops that aren't yet on MPS):
       export PYTORCH_ENABLE_MPS_FALLBACK=1

  3) (Optional) Configure Kaggle and Wandb credentials:
       # ~/.kaggle/kaggle.json  (chmod 600)
       wandb login

  4) Run the verification script:
       python scripts/verify_mps.py

EOF
