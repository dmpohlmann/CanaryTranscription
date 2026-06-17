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
- **`transcribe_runpod.py`** — a GPU-tuned variant for RunPod pods. Hard-fails without CUDA, runs Whisper in `float16` with `sdpa` attention and `batch_size=24` batched decoding, and adds a `--url` flag to download audio from a URL (skipping the Jupyter upload step). The pipeline logic is otherwise a verbatim copy of `transcribe.py` (its `--word-snap` word pass uses large-v3 since the GPU handles it; the CPU script uses a small model). Note: `flash_attention_2` is **not** used — word-level timestamps need cross-attention weights that flash-attn can't return, so `sdpa` is required.

The pipeline (both scripts):

1. **ASR** — Loads `openai/whisper-large-v3` via `transformers.pipeline`, resamples audio to 16 kHz with `librosa`, and transcribes with chunk-level timestamps (punctuated text), cached to `output/<filename>_asr_cache.json`. With `--word-snap`, a **second** pass on a small fast model (`openai/whisper-base`) produces per-word timestamps, cached to `output/<filename>_asr_words.json` (large-v3 word-mode is pathologically slow on CPU, so the timestamp pass uses a small model). Both caches let re-runs skip the expensive ASR step.
2. **Diarization** — Runs `pyannote/speaker-diarization-3.1` (requires a Hugging Face token with accepted model license) to identify who spoke when.
3. **Merge** — Two modes. By default (`merge_chunks_with_diarization`), each punctuated chunk is assigned to the speaker who covers most of it (smoother, but a turn can start mid-sentence). With `--word-snap` (`merge_punctuated_with_diarization`), chunks are split at speaker-change boundaries using the word-level timestamps, so turns snap to who is actually speaking while keeping the punctuated text. Both drop whole chunks with empty text or no diarised speaker (`UNKNOWN`) — non-speech regions where Whisper hallucinates filler like "Thank you for watching"; word-snap keeps real words within a speech chunk by falling back to its dominant speaker.
4. **Name detection** — Regex-based heuristics in `detect_speaker_names` scan the transcript for self-introductions (e.g., "my name is…", "Senator…", "…for the record") to replace generic `SPEAKER_XX` labels with real names.

Output is written to `output/<filename>_diarised.txt`.

## Key conventions

- Audio files go in `audio/`, transcripts go in `output/` — both are gitignored
- ASR caches (`_asr_cache.json` for punctuated text, `_asr_words.json` for word timestamps) enable iterating on diarization/formatting without re-running the slow transcription step; delete a cache file to force that pass to re-run
- Device selection is automatic: CUDA GPU if available, otherwise CPU
- The script uses `argparse` for CLI args; `--hf-token` or `HF_TOKEN` env var is required; `--word-snap` opts into word-level boundary snapping (default is smoother chunk-level boundaries)
- M4A/AAC audio needs `ffmpeg`; on a no-sudo box, `pip install imageio-ffmpeg` and symlink its binary into `venv/bin/ffmpeg`. `run.sh` wraps a local run (puts `ffmpeg` on PATH, sources `HF_TOKEN` from `hf auth login`)

## Dependencies

Two requirements files:
- `requirements.txt` — the GPU/RunPod set: `transformers`, `accelerate`, `librosa`, `pyannote.audio`, `flash-attn`.
- `requirements-cpu.txt` — CPU-only machines (no NVIDIA GPU). Same set minus `flash-attn`, plus explicit `torch`/`torchaudio`/`soundfile`, with guidance for installing the lean CPU PyTorch wheel. Use this for local desktop and the future Windows runs.

Notes:
- `torch` is not listed explicitly — it is pulled in transitively by `transformers`/`pyannote.audio`. For a specific CUDA/CPU build, install `torch` first from the appropriate PyTorch index before `pip install -r requirements.txt`.
- `flash-attn` is listed in `requirements.txt` but **no longer used** — `transcribe_runpod.py` switched to `sdpa` because word-level timestamps need cross-attention weights flash-attn can't return. It is GPU-only and fails to build on CPU, so omit it for CPU setups (it's absent from `requirements-cpu.txt`).
- `requirements-cpu.txt` includes `imageio-ffmpeg`, which ships a static `ffmpeg` binary for decoding M4A/AAC without a system ffmpeg install.
- The pipeline pre-decodes audio with `librosa` and includes monkeypatches to bypass `torchcodec` (in the transformers ASR pipeline) and pyannote's `AudioDecoder`. `ffmpeg` is required so `librosa` can decode M4A/AAC and recommended for MP3.
