#!/usr/bin/env bash
# Convenience wrapper for local CPU runs of transcribe.py (Linux/WSL).
#
# It handles the two environment details a bare `python transcribe.py` needs on
# a CPU-only setup here:
#   1. ffmpeg on PATH — the bundled imageio-ffmpeg binary (symlinked into
#      venv/bin/ffmpeg), so librosa can decode M4A/AAC and other compressed audio.
#   2. HF_TOKEN — sourced from the token stored by `hf auth login`, so the
#      pyannote diarisation models can be downloaded. The token is never echoed
#      or placed on a command line.
#
# Usage:
#   ./run.sh audio/your_recording.m4a
#
# (On Windows, run transcribe.py directly with HF_TOKEN set — this bash wrapper
#  is for the Linux/WSL dev environment.)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "error: venv not found at $VENV — create it first (see README/CLAUDE.md)." >&2
  exit 1
fi

# 1. Put venv/bin first so the bundled ffmpeg symlink is discoverable.
export PATH="$VENV/bin:$PATH"

# 2. Use an explicit HF_TOKEN if already set; otherwise fall back to the stored
#    `hf auth login` token.
if [ -z "${HF_TOKEN:-}" ]; then
  if HF_TOKEN="$("$VENV/bin/hf" auth token 2>/dev/null)" && [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
  else
    echo "error: no HF_TOKEN set and no stored token found." >&2
    echo "       run: $VENV/bin/hf auth login" >&2
    exit 1
  fi
fi

exec "$VENV/bin/python" "$REPO_DIR/transcribe.py" "$@"
