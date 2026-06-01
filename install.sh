#!/usr/bin/env bash
# install.sh — one-command setup for the MKD fine-tuning platform.
#
# Handles automatically:
#   - CUDA version detection (11, 12, 13)
#   - Correct PyTorch wheel selection
#   - Axolotl + flash-attn install
#   - Gradio UI dependencies
#   - ft CLI symlink + FT_PROJECT env var
#
# Requirements:
#   - Ubuntu 22.04+  (or WSL2)
#   - NVIDIA driver already installed  (run: nvidia-smi)
#   - Python 3.10+

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; CYAN='\033[36m'; RESET='\033[0m'
info()  { echo -e "${CYAN}ℹ $*${RESET}"; }
ok()    { echo -e "${GREEN}✓ $*${RESET}"; }
warn()  { echo -e "${YELLOW}⚠ $*${RESET}"; }
err()   { echo -e "${RED}✗ $*${RESET}" >&2; }

echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │   MKD Fine-Tuning Platform — Installer      │"
echo "  └─────────────────────────────────────────────┘"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 >/dev/null; then
  err "python3 not found. Install Python 3.10+ and re-run."; exit 1
fi
PY_MAJ=$(python3 -c 'import sys;print(sys.version_info.major)')
PY_MIN=$(python3 -c 'import sys;print(sys.version_info.minor)')
if [ "$PY_MAJ" -lt 3 ] || { [ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt 10 ]; }; then
  err "Python ≥ 3.10 required (got $PY_MAJ.$PY_MIN)"; exit 1
fi
ok "Python $PY_MAJ.$PY_MIN"

# ── Venv ──────────────────────────────────────────────────────────────────────
SHARED_VENV="$HOME/fine-tune/venv"
if [ -d "$SHARED_VENV" ] && [ -f "$SHARED_VENV/bin/activate" ]; then
  VENV_DIR="$SHARED_VENV"
  info "Using existing venv at $VENV_DIR"
else
  VENV_DIR="$ROOT/.venv"
  info "Creating venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel -q
ok "Venv ready: $VENV_DIR"

# ── CUDA detection ────────────────────────────────────────────────────────────
CUDA_MAJOR=0
if command -v nvidia-smi >/dev/null; then
  CUDA_RAW=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}' || true)
  CUDA_MAJOR="${CUDA_RAW%%.*}"
  info "Detected CUDA $CUDA_RAW (driver reports)"
else
  warn "nvidia-smi not found — installing CPU-only torch (dataset prep only, no training)"
fi

# ── PyTorch ───────────────────────────────────────────────────────────────────
info "Installing PyTorch..."
case "$CUDA_MAJOR" in
  13|12) PT_INDEX="https://download.pytorch.org/whl/cu128" ;;  # cu128 works on CUDA 12 & 13
  11)    PT_INDEX="https://download.pytorch.org/whl/cu118" ;;
  *)     PT_INDEX="https://download.pytorch.org/whl/cpu"   ;;
esac
info "Using PyTorch index: $PT_INDEX"
pip install torch torchvision torchaudio --index-url "$PT_INDEX" -q
ok "PyTorch installed"

# ── Core platform deps ────────────────────────────────────────────────────────
info "Installing core dependencies..."
pip install -q \
  huggingface_hub \
  datasets \
  transformers \
  accelerate \
  peft \
  trl \
  bitsandbytes \
  sentencepiece \
  protobuf \
  pandas \
  pyarrow \
  datasketch \
  requests \
  pyyaml \
  tensorboard \
  gradio
ok "Core dependencies installed"

# ── Axolotl ───────────────────────────────────────────────────────────────────
info "Installing Axolotl (training backend)..."
if python3 -c "import axolotl" 2>/dev/null; then
  ok "Axolotl already installed"
else
  # Install from source to avoid missing-file bugs in PyPI releases
  pip install -q "git+https://github.com/axolotl-ai-cloud/axolotl.git" || {
    warn "GitHub install failed, trying PyPI..."
    pip install -q axolotl
  }
  ok "Axolotl installed"
fi

# ── Flash Attention (optional but recommended) ────────────────────────────────
info "Installing FlashAttention-2 (speeds up training ~2×)..."
if python3 -c "import flash_attn" 2>/dev/null; then
  ok "FlashAttention already installed"
elif [ "$CUDA_MAJOR" -ge 11 ] 2>/dev/null; then
  pip install -q packaging ninja
  pip install -q flash-attn --no-build-isolation || {
    warn "FlashAttention build failed — training will work without it (slightly slower)"
    warn "To retry manually: pip install flash-attn --no-build-isolation"
  }
else
  warn "Skipping FlashAttention (no CUDA detected)"
fi

# ── ft CLI ────────────────────────────────────────────────────────────────────
mkdir -p "$HOME/.local/bin"
chmod +x "$ROOT/scripts/ft"
ln -sf "$ROOT/scripts/ft" "$HOME/.local/bin/ft"
ok "Linked $HOME/.local/bin/ft → $ROOT/scripts/ft"

# ── Shell profile ─────────────────────────────────────────────────────────────
PROFILE_SNIPPET="
# >>> ft fine-tuning platform >>>
export FT_PROJECT=\"$ROOT\"
export PATH=\"\$HOME/.local/bin:\$PATH\"
# <<< ft fine-tuning platform <<<"

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  if [ -f "$rc" ] && ! grep -q "ft fine-tuning platform" "$rc"; then
    printf "%s\n" "$PROFILE_SNIPPET" >> "$rc"
    ok "Added FT_PROJECT/PATH to $rc"
  fi
done

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │              Install Summary                 │"
echo "  ├─────────────────────────────────────────────┤"
python3 -c "import torch; print(f'  │  PyTorch:      {torch.__version__:<28}│')"
python3 -c "
import torch
avail = torch.cuda.is_available()
name  = torch.cuda.get_device_name(0) if avail else 'N/A'
print(f'  │  CUDA avail:   {str(avail):<28}│')
print(f'  │  GPU:          {name[:28]:<28}│')
"
python3 -c "import axolotl; print(f'  │  Axolotl:      OK{\"\":<28}│')" 2>/dev/null || \
  echo "  │  Axolotl:      FAILED (check above)         │"
python3 -c "import gradio as gr; print(f'  │  Gradio:       {gr.__version__:<28}│')"
echo "  └─────────────────────────────────────────────┘"
echo ""
ok "Install complete!"
echo ""
echo "  Open a new terminal (or: source ~/.bashrc), then:"
echo "    ft --help          ← CLI"
echo "    ft ui              ← Web UI at http://localhost:7860"
echo ""
echo "  First run: read guides/00_quickstart.md"
echo ""
