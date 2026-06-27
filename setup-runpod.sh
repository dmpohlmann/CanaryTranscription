#!/usr/bin/env bash
# Set up CanaryTranscription on a RunPod PyTorch pod (e.g. RTX 4090).
#
# Assumes the official "RunPod PyTorch 2.x" template, which already ships
# torch + CUDA. The venv and Hugging Face cache live under the network volume
# ($VOLUME, default /workspace), so they persist across pod restarts — run this
# once per volume and subsequent pods just `source` the venv.
#
# Usage:
#   bash setup-runpod.sh
#   source /workspace/venv/bin/activate
#   python transcribe_runpod.py --url <audio-url>     # or a local path
set -euo pipefail

VOLUME="${VOLUME:-/workspace}"
VENV="$VOLUME/venv"
HF_CACHE="$VOLUME/hf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[setup] Installing ffmpeg (for M4A/AAC decoding)..."
apt-get update -qq && apt-get install -y -qq ffmpeg

echo "[setup] Creating venv at $VENV (inheriting the image's torch + CUDA)..."
# --system-site-packages so the venv sees the pre-installed torch/CUDA from the
# RunPod image; only the extra packages below get installed into the volume.
python -m venv --system-site-packages "$VENV"

echo "[setup] Persisting HF_HOME=$HF_CACHE so large-v3 downloads only once..."
mkdir -p "$HF_CACHE"
# Export HF_HOME whenever the venv is activated, so the model cache lands on the
# volume rather than the ephemeral container filesystem.
grep -q "export HF_HOME=" "$VENV/bin/activate" \
  || echo "export HF_HOME=$HF_CACHE" >> "$VENV/bin/activate"

echo "[setup] Installing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements-runpod.txt"

echo
echo "[setup] Done. Next:"
echo "  source $VENV/bin/activate"
echo "  export HF_TOKEN=hf_xxx          # token with pyannote license accepted"
echo "  python transcribe_runpod.py --url <audio-url>"
