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

This installs:

| Package | Purpose |
|---|---|
| `transformers` | Hugging Face library — loads and runs the ASR model |
| `torch` | PyTorch — the underlying deep learning framework |
| `torchaudio` | Audio processing utilities for PyTorch |
| `accelerate` | Optimises model performance on available hardware |

> First-time installation may take several minutes depending on connection speed.

---

## Running a transcription

> *[To be completed as the script is developed]*

---

## Project structure

```
CanaryTranscription/
├── venv/                  # Virtual environment (not committed to git)
├── audio/                 # Place your MP3 files here
├── output/                # Transcripts saved here
├── transcribe.py          # Main transcription script
├── requirements.txt       # Python dependencies
├── .gitignore
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

> AMD GPU support via ROCm is possible but requires a different PyTorch installation — see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for details.

---

## Useful references

- [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [OpenAI Whisper (original model)](https://github.com/openai/whisper)

---

*Guide developed February 2026. Sections marked [To be completed] will be updated as the project develops.*
