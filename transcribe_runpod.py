"""
RunPod-optimised transcription script.

Identical pipeline to transcribe.py (Whisper large-v3 + pyannote diarization)
but tuned for GPU pods: float16 inference, flash attention, batched decoding,
and a --url flag so you can skip the Jupyter upload dance.
"""

import os
import json
import sys
import torch
import argparse
import re
import librosa
import numpy as np
import urllib.request
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.core import Segment
from pathlib import Path

# Prevent broken torchcodec from crashing the transformers ASR pipeline.
# We pre-decode audio with librosa, so torchcodec is never needed.
# Must patch the ASR module directly — it holds its own imported reference.
from transformers.pipelines import automatic_speech_recognition as _asr_module
_asr_module.is_torchcodec_available = lambda: False

# PyTorch 2.6+ defaults torch.load to weights_only=True, which blocks
# older model checkpoints that contain classes like TorchVersion.
# Allowlist them so pyannote can load its segmentation model.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    if seconds is None:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_speaker_for_segment(diarization, start, end):
    """Find the dominant speaker for a given time segment by overlap."""
    overlaps = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(turn.start, start)
        overlap_end = min(turn.end, end)
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            overlaps[speaker] = overlaps.get(speaker, 0) + overlap
    if not overlaps:
        return "UNKNOWN"
    return max(overlaps, key=overlaps.get)


def merge_transcript_with_diarization(transcript_chunks, diarization):
    """Merge Whisper timestamp chunks with pyannote speaker labels."""
    merged = []
    for chunk in transcript_chunks:
        start = chunk["timestamp"][0]
        end = chunk["timestamp"][1]
        if start is None or end is None:
            continue
        speaker = get_speaker_for_segment(diarization, start, end)
        merged.append({
            "start": start,
            "end": end,
            "text": chunk["text"].strip(),
            "speaker": speaker,
        })
    return merged


def detect_speaker_names(merged_segments):
    """
    Scan transcript segments for speaker self-introductions and return a
    mapping of speaker ID to detected name.

    Patterns targeted at Australian Senate committee hearings:
      - "my name is Jane Smith"
      - "Jane Smith for the record"
      - "I'm Jane Smith"
      - "Senator Jane Smith"
      - "Professor / Doctor / Secretary Jane Smith"
    """
    speaker_names = {}
    name_patterns = [
        r"(?:my name is|i(?:'m| am))\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",
        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})[,.]?\s+for the record",
        r"(?:senator|chair|secretary|professor|doctor|dr\.?)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})",
    ]
    for segment in merged_segments:
        speaker = segment["speaker"]
        if speaker in speaker_names:
            continue
        for pattern in name_patterns:
            match = re.search(pattern, segment["text"], re.IGNORECASE)
            if match:
                speaker_names[speaker] = match.group(1).title()
                break
    return speaker_names


def format_transcript(merged_segments, speaker_names):
    """
    Format merged segments into a readable transcript, grouping consecutive
    segments from the same speaker and printing a timestamp at each change.
    """
    lines = []
    current_speaker = None
    current_text = []
    current_start = None

    for seg in merged_segments:
        speaker_id = seg["speaker"]
        display_name = speaker_names.get(speaker_id, speaker_id)

        if display_name != current_speaker:
            if current_speaker is not None and current_text:
                lines.append(
                    f"\n[{format_time(current_start)}] {current_speaker}:\n"
                    + " ".join(current_text)
                )
            current_speaker = display_name
            current_text = [seg["text"]]
            current_start = seg["start"]
        else:
            current_text.append(seg["text"])

    # Flush final speaker block
    if current_speaker is not None and current_text:
        lines.append(
            f"\n[{format_time(current_start)}] {current_speaker}:\n"
            + " ".join(current_text)
        )

    return "\n".join(lines)


def download_audio(url, dest_dir):
    """Download an audio file from a URL into dest_dir. Returns the local path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    # Extract filename from URL, falling back to a default
    filename = Path(urllib.request.urlparse(url).path).name or "download.mp3"
    dest_path = dest_dir / filename

    if dest_path.exists():
        print(f"File already exists: {dest_path} — skipping download.")
        return str(dest_path)

    print(f"Downloading: {url}")
    print(f"         to: {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Download complete ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return str(dest_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def transcribe(audio_path: str, hf_token: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. This script is designed for RunPod GPU pods. "
            "Use transcribe.py for CPU-based local transcription."
        )

    device = "cuda"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    input_filename = Path(audio_path).stem
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    cache_path = output_dir / f"{input_filename}_asr_cache.json"

    # ── Step 1: Transcription (with checkpointing) ────────────────────────────
    if cache_path.exists():
        print(f"\n[1/4] Found cached transcription — skipping ASR step.")
        print(f"      Delete {cache_path} to force re-transcription.")
        with open(cache_path, "r", encoding="utf-8") as f:
            asr_result = json.load(f)
    else:
        print("\n[1/4] Loading Whisper Large V3 model (float16 + flash attention)...")
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=device,
            torch_dtype=torch.float16,
            model_kwargs={"attn_implementation": "flash_attention_2"},
        )

        print(f"[1/4] Loading audio file: {audio_path}")
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
        print(f"[1/4] Transcribing ({len(audio_array)/sampling_rate/3600:.1f} hours of audio)...")
        asr_result = asr_pipe(
            {"raw": audio_array, "sampling_rate": sampling_rate},
            return_timestamps=True,
            batch_size=24,
            generate_kwargs={"language": "en"},
        )

        print(f"[1/4] Transcription complete — saving cache to {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(asr_result, f, ensure_ascii=False, indent=2)

    # ── Step 2: Diarisation ───────────────────────────────────────────────────
    print("\n[2/4] Loading speaker diarisation model...")
    try:
        # pyannote.audio 4.x
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        # pyannote.audio 3.x
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    diarization_pipeline.to(torch.device(device))

    # Load audio with librosa and pass as in-memory waveform to avoid
    # pyannote's AudioDecoder (which requires torchcodec + FFmpeg .so files).
    print(f"[2/4] Loading audio for diarisation: {audio_path}")
    waveform_np, sr = librosa.load(audio_path, sr=16000, mono=True)
    waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, samples)
    diarization_input = {
        "waveform": waveform_tensor,
        "sample_rate": sr,
        "uri": Path(audio_path).stem,
    }

    print("[2/4] Running diarisation (this may take a while for long audio)...")
    diarization = diarization_pipeline(diarization_input)

    # pyannote 4.x returns a DiarizeOutput wrapper; unwrap to the Annotation
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    print("\n[3/4] Merging transcript with speaker labels...")
    chunks = asr_result.get("chunks", [])
    merged = merge_transcript_with_diarization(chunks, diarization)

    # ── Step 4: Name detection and output ─────────────────────────────────────
    print("\n[4/4] Detecting speaker names from transcript...")
    speaker_names = detect_speaker_names(merged)

    if speaker_names:
        print("Detected speaker name mappings:")
        for speaker_id, name in speaker_names.items():
            print(f"  {speaker_id} → {name}")
    else:
        print("No names automatically detected — speakers will be labelled SPEAKER_00, SPEAKER_01 etc.")
        print("You can rename them manually in the output file.")

    transcript = format_transcript(merged, speaker_names)

    output_path = output_dir / f"{input_filename}_diarised.txt"
    output_path.write_text(transcript, encoding="utf-8")

    print(f"\nTranscription complete. Saved to: {output_path}")
    print("\n--- Preview (first 1000 characters) ---")
    print(transcript[:1000])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file and label speakers (RunPod GPU version)."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to the audio file (e.g. audio/hearing.mp3). "
             "Optional if --url is provided.",
    )
    parser.add_argument(
        "--url",
        help="Download audio from this URL instead of using a local file path.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token. Can also be set as the HF_TOKEN environment variable "
             "(recommended — avoids token appearing in terminal history).",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError(
            "No Hugging Face token provided. "
            "Use --hf-token YOUR_TOKEN or set the HF_TOKEN environment variable."
        )

    if args.url:
        audio_path = download_audio(args.url, "audio")
    elif args.audio:
        audio_path = args.audio
    else:
        parser.error("Provide either an audio file path or --url.")

    transcribe(audio_path, args.hf_token)
