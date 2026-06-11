# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CanaryTranscription is a local audio transcription pipeline that combines OpenAI Whisper (via Hugging Face `transformers`) with pyannote speaker diarization to produce speaker-labelled transcripts. It is designed for Australian government use (DCCEEW), processing recordings such as Senate committee hearings entirely on-device.

## Commands

```bash
# Setup (Python 3.11 required for PyTorch compatibility)
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run transcription
python transcribe.py audio/recording.mp3 --hf-token YOUR_TOKEN
# Or with environment variable (preferred — avoids token in shell history)
set HF_TOKEN=YOUR_TOKEN
python transcribe.py audio/recording.mp3
```

There are no tests, linting, or build steps — this is a single-script tool.

## Architecture

There are two entry-point scripts that share an identical 4-step pipeline and helper functions, differing only in device handling and ASR configuration:

- **`transcribe.py`** — the portable version. Auto-selects CUDA if available, otherwise CPU. Use this for local/desktop runs.
- **`transcribe_runpod.py`** — a GPU-tuned variant for RunPod pods. Hard-fails without CUDA, runs Whisper in `float16` with `flash_attention_2` and `batch_size=24` batched decoding, and adds a `--url` flag to download audio from a URL (skipping the Jupyter upload step). The pipeline logic is otherwise a verbatim copy of `transcribe.py`.

The pipeline (both scripts):

1. **ASR** — Loads `openai/whisper-large-v3` via `transformers.pipeline`, resamples audio to 16 kHz with `librosa`, and transcribes with word-level timestamps. Results are cached to `output/<filename>_asr_cache.json` so re-runs skip this expensive step.
2. **Diarization** — Runs `pyannote/speaker-diarization-3.1` (requires a Hugging Face token with accepted model license) to identify who spoke when.
3. **Merge** — Aligns Whisper's timestamped chunks with pyannote's speaker turns using overlap-based matching (`get_speaker_for_segment`).
4. **Name detection** — Regex-based heuristics in `detect_speaker_names` scan the transcript for self-introductions (e.g., "my name is…", "Senator…", "…for the record") to replace generic `SPEAKER_XX` labels with real names.

Output is written to `output/<filename>_diarised.txt`.

## Key conventions

- Audio files go in `audio/`, transcripts go in `output/` — both are gitignored
- The ASR cache (`_asr_cache.json`) enables iterating on diarization/formatting without re-running the slow transcription step; delete it to force re-transcription
- Device selection is automatic: CUDA GPU if available, otherwise CPU
- The script uses `argparse` for CLI args; `--hf-token` or `HF_TOKEN` env var is required

## Dependencies

Two requirements files:
- `requirements.txt` — the GPU/RunPod set: `transformers`, `accelerate`, `librosa`, `pyannote.audio`, `flash-attn`.
- `requirements-cpu.txt` — CPU-only machines (no NVIDIA GPU). Same set minus `flash-attn`, plus explicit `torch`/`torchaudio`/`soundfile`, with guidance for installing the lean CPU PyTorch wheel. Use this for local desktop and the future Windows runs.

Notes:
- `torch` is not listed explicitly — it is pulled in transitively by `transformers`/`pyannote.audio`. For a specific CUDA/CPU build, install `torch` first from the appropriate PyTorch index before `pip install -r requirements.txt`.
- `flash-attn` is **GPU-only** and only used by `transcribe_runpod.py`. It will fail to build on a CPU-only machine and is not needed for `transcribe.py` — install the rest manually (or comment it out) for CPU setups.
- The pipeline pre-decodes audio with `librosa` and includes monkeypatches to bypass `torchcodec` (in the transformers ASR pipeline) and pyannote's `AudioDecoder`. `ffmpeg` is still recommended so `librosa` can decode MP3 and other compressed formats.
