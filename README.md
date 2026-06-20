# Audio Transcription with Hugging Face — Setup Guide

A step-by-step guide to building a local audio transcription pipeline using open-source AI models from Hugging Face. Developed for use within DCCEEW as a reusable, portable pattern for transcribing recorded audio files.

---

## Background

This project uses the [Hugging Face `transformers`](https://huggingface.co/docs/transformers) library to run a speech-to-text (automatic speech recognition, or ASR) model locally on your machine — meaning audio never leaves your environment and there are no per-minute API costs.

The model used is sourced from the [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), which benchmarks models by word error rate (WER) — the lower the WER, the more accurate the transcription.

### How the audio was generated
> *[To be completed by David — describe the recording setup, source, format, duration etc.]*

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10 or 11 | Guide written for Windows; steps are similar on Mac/Linux |
| Python 3.11 | See note below — 3.11 recommended over newer versions for ML compatibility |
| Git | For version control and cloning to other machines |
| ~5 GB disk space | For model weights downloaded on first run |

### Why Python 3.11?

PyTorch — the underlying deep learning framework — takes several months to add support for new Python releases. Python 3.14 (released late 2025) is not yet reliably supported. Python 3.11 is the most stable and widely tested version for machine learning work as of early 2026.

You can install Python 3.11 alongside an existing Python installation without conflict.

1. Download Python 3.11 from [python.org/downloads](https://www.python.org/downloads/)
2. During installation, **do not** check "Add to PATH" (to avoid overriding your existing Python)
3. Verify the install by running `py -3.11 --version` in Command Prompt

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CanaryTranscription.git
cd CanaryTranscription
```

Or, if setting up from scratch on a new machine:

```bash
cd C:\Users\<username>\repositories
git clone https://github.com/<your-username>/CanaryTranscription.git
cd CanaryTranscription
```

### 2. Create a virtual environment

A virtual environment isolates this project's Python packages from the rest of your system — this avoids version conflicts and keeps the setup reproducible.

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)` at the start, confirming the environment is active.

> **Note:** The `venv` folder is excluded from version control (see `.gitignore`). On a new machine you recreate it using the steps above, then install packages from `requirements.txt`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **CPU-only machine (no NVIDIA GPU)?** Use `pip install -r requirements-cpu.txt`
> instead — it omits the GPU-only `flash-attn` package (which won't build on CPU)
> and installs a CPU-compatible PyTorch.

This installs:

| Package | Purpose |
|---|---|
| `transformers` | Hugging Face library — loads and runs the Whisper ASR model |
| `accelerate` | Optimises model performance on available hardware |
| `librosa` | Decodes and resamples audio to 16 kHz before transcription |
| `pyannote.audio` | Speaker diarisation — identifies who spoke when |
| `flash-attn` | **GPU-only** fast attention, used by `transcribe_runpod.py` |

> `torch` (PyTorch) is not pinned directly — it is pulled in automatically as a dependency. To control the CUDA vs CPU build, install `torch` first from the [PyTorch index](https://pytorch.org/get-started/locally/), then run the command above.

> **`flash-attn` is GPU-only and will fail to build on a CPU-only machine.** For a CPU setup, install the other packages manually (or remove the `flash-attn` line) — it is only needed by the RunPod GPU script, not by `transcribe.py`.

> First-time installation may take several minutes depending on connection speed.

---

## Running a transcription

Speaker diarisation requires a free Hugging Face access token with the
`pyannote/speaker-diarization-3.1` model licence accepted (visit the model page
on huggingface.co and click *Agree* once while signed in).

```bash
# Pass the token on the command line...
python transcribe.py audio/recording.mp3 --hf-token YOUR_TOKEN

# ...or set it as an environment variable (preferred — keeps it out of shell history)
export HF_TOKEN=YOUR_TOKEN        # Windows: set HF_TOKEN=YOUR_TOKEN
python transcribe.py audio/recording.mp3
```

The script runs four steps — ASR (Whisper), diarisation (pyannote), merge, and
name detection — and writes the result to `output/<filename>_diarised.txt`.

### Turn boundaries: `--word-snap`

By default, each ~5-second Whisper chunk is attributed to whichever speaker talks
most of it. This reads smoothly, but a turn can start mid-sentence when speakers
change inside a chunk. Add `--word-snap` to split chunks at the exact speaker
change using word-level timestamps:

```bash
python transcribe.py audio/recording.mp3 --word-snap
```

This gives more accurate turn boundaries (and keeps the punctuated text), at the
cost of a second, fast ASR pass and choppier output during rapid back-and-forth.
Recommended when precise attribution matters (e.g. who said what in a hearing).

### Lowercase output: `--restore-punct`

Whisper sometimes transcribes a recording almost entirely lowercase and
unpunctuated (it's audio-dependent). `--restore-punct` repairs punctuation and
capitalisation — including proper nouns — with a small on-device model:

```bash
python transcribe.py audio/recording.wav --word-snap --restore-punct
```

It auto-skips transcripts that are already well punctuated, so it's safe to add
to every command in a batch — it only fires where it's needed. Expect occasional
minor glitches (a miscased word, a stray sentence break); review the output.

### Known speaker count: `--speakers`

If you know how many people are in the recording, tell the diariser so it doesn't
invent extra speakers:

```bash
python transcribe.py audio/interview.wav --speakers 2
```

`--speakers 2` is the common case for a one-on-one interview. Without it, pyannote
estimates the count and can occasionally split one person into two (or add a
phantom speaker from background noise).

> **ASR cache:** the slow transcription step is cached to
> `output/<filename>_asr_cache.json`. Re-running the same file reuses the cache so
> you can iterate on diarisation and formatting without re-transcribing. Delete the
> cache file to force a fresh transcription.

### Running on Linux / WSL

The setup commands above are written for Windows. On Linux or WSL the equivalents are:

```bash
python3 -m venv venv          # or: uv venv --python 3.11 venv
source venv/bin/activate
pip install -r requirements-cpu.txt
```

`librosa` needs `ffmpeg` to decode **M4A/AAC** (and it's recommended for MP3 too).
If you can't install system ffmpeg (e.g. no sudo), the bundled static binary works:

```bash
pip install imageio-ffmpeg
ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" venv/bin/ffmpeg
```

GPU acceleration under WSL additionally requires the NVIDIA driver installed on
the **Windows** host — see *GPU acceleration* below.

**Convenience wrapper:** `./run.sh audio/recording.m4a` runs `transcribe.py` with
`ffmpeg` on PATH and `HF_TOKEN` sourced from your stored `hf auth login` token, so
you don't have to set them each time.

---

## Project structure

```
CanaryTranscription/
├── venv/                  # Virtual environment (not committed to git)
├── audio/                 # Place your MP3 files here (not committed to git)
├── output/                # Transcripts + ASR cache saved here (not committed to git)
├── transcribe.py          # Main transcription script (CPU or GPU, auto-detected)
├── transcribe_runpod.py   # GPU-tuned variant for RunPod pods (float16 + flash attention, --url download)
├── run.sh                 # Convenience wrapper for local runs (Linux/WSL)
├── requirements.txt       # Python dependencies (GPU / RunPod)
├── requirements-cpu.txt   # Python dependencies (CPU-only machines)
├── .gitignore
├── CLAUDE.md              # Guidance for Claude Code
└── README.md
```

---

## Porting to another machine

1. Clone the repository
2. Install Python 3.11 if not already present
3. Create a fresh virtual environment: `py -3.11 -m venv venv`
4. Activate it: `venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`

### GPU acceleration

If your machine has an NVIDIA GPU, PyTorch can use CUDA to significantly speed up transcription. After completing the standard setup, check whether CUDA is available by running:

```python
import torch
print(torch.cuda.is_available())
```

If this returns `True`, the transcription script will automatically use the GPU. If `False`, it will fall back to CPU (slower but fully functional).

> **WSL2:** CUDA inside WSL requires the NVIDIA driver to be installed on the **Windows host** (not inside WSL). Once it is, `nvidia-smi` becomes available inside WSL and `torch.cuda.is_available()` returns `True`. If `nvidia-smi` is missing inside WSL, the Windows-side driver is not installed or not WSL-enabled, and PyTorch will only see the CPU.

> AMD GPU support via ROCm is possible but requires a different PyTorch installation — see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for details.

---

## Useful references

- [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [OpenAI Whisper (original model)](https://github.com/openai/whisper)

---

*Guide developed February 2026. Sections marked [To be completed] will be updated as the project develops.*
