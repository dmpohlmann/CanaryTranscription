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

Everything lives in `transcribe.py`, which runs a 4-step pipeline:

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

Core: `transformers`, `torch`, `torchaudio`, `librosa`, `pyannote.audio`, `accelerate`

Note: `requirements.txt` is currently empty — dependencies must be installed manually or the file needs updating.
