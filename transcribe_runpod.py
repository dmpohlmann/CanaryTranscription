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
import bisect
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

# On-device punctuation + truecasing model (ONNX, CPU-friendly) used by
# --restore-punct to repair recordings Whisper transcribed lowercase/unpunctuated.
PUNCT_RESTORE_MODEL = "pcs_en"


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    if seconds is None:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _flatten_turns(diarization):
    """pyannote Annotation -> list of (start, end, speaker) turns, sorted."""
    turns = [(turn.start, turn.end, spk)
             for turn, _, spk in diarization.itertracks(yield_label=True)]
    turns.sort()
    return turns


def _speaker_at(turns, t):
    """Speaker talking at instant t, or UNKNOWN if no turn covers it."""
    for start, end, spk in turns:
        if start <= t < end:
            return spk
    return "UNKNOWN"


def _dominant_speaker(turns, start, end):
    """Speaker with the most overlap of [start, end], or UNKNOWN if none."""
    overlaps = {}
    for s, e, spk in turns:
        lo, hi = max(s, start), min(e, end)
        if hi > lo:
            overlaps[spk] = overlaps.get(spk, 0) + (hi - lo)
    if not overlaps:
        return "UNKNOWN"
    return max(overlaps, key=overlaps.get)


def merge_chunks_with_diarization(punct_chunks, diarization):
    """Default merge: assign each punctuated Whisper chunk to the speaker who
    covers most of it, keeping the chunk whole. Chunks with empty text or no
    diarised speaker (silence/hallucination) are dropped.

    Boundaries follow Whisper's ~5 s chunks, so a chunk spanning a speaker change
    is attributed to whichever speaker dominates it — smoother output, but a turn
    can start mid-sentence. Pass --word-snap for word-level boundary snapping.
    """
    turns = _flatten_turns(diarization)
    segments = []
    for chunk in punct_chunks:
        ts = chunk.get("timestamp") or (None, None)
        cs, ce = ts[0], ts[1]
        if cs is None:
            continue
        if ce is None or ce <= cs:
            ce = cs + 0.20
        text = chunk["text"].strip()
        if not text:
            continue
        speaker = _dominant_speaker(turns, cs, ce)
        if speaker == "UNKNOWN":
            continue
        segments.append({"start": cs, "end": ce, "text": text, "speaker": speaker})
    return segments


def merge_punctuated_with_diarization(punct_chunks, word_chunks, diarization):
    """Hybrid merge: keep the punctuated chunk-level text, but cut each chunk at
    pyannote speaker boundaries using the word-level timestamps to find where
    inside the chunk each boundary falls.

    Two ASR passes feed this:
      * punct_chunks — chunk-level pass (return_timestamps=True): ~5 s chunks of
        properly punctuated, capitalised text, but with coarse boundaries.
      * word_chunks  — word-level pass (return_timestamps="word"): one bare token
        per word, but with precise per-word start times.

    A chunk wholly inside one speaker turn is emitted unchanged (punctuation
    intact). A chunk that straddles a speaker change is split: the count of
    word-level words spoken before the boundary time gives the fraction of the
    chunk's words to assign to the first speaker, so the punctuation-bearing text
    is divided at the right place rather than reconstructed from bare tokens.
    Sub-segments with no speaker (UNKNOWN) are dropped — this also removes
    Whisper's hallucinated filler over trailing silence.
    """
    turns = _flatten_turns(diarization)
    word_starts = sorted(
        w["timestamp"][0] for w in word_chunks
        if w.get("timestamp") and w["timestamp"][0] is not None
    )

    segments = []
    for chunk in punct_chunks:
        ts = chunk.get("timestamp") or (None, None)
        cs, ce = ts[0], ts[1]
        if cs is None:
            continue
        if ce is None or ce <= cs:
            ce = cs + 0.20
        words = chunk["text"].split()
        if not words:
            continue
        n = len(words)

        # Speaker covering most of the chunk. If nobody is diarised as speaking
        # anywhere in the chunk, it is non-speech (silence/noise) where Whisper
        # hallucinates filler — drop the whole chunk. Otherwise the chunk is real
        # speech, so its words are never dropped, only attributed.
        dominant = _dominant_speaker(turns, cs, ce)
        if dominant == "UNKNOWN":
            continue

        # Speaker-change times strictly inside the chunk are the split points.
        cuts = sorted({t for s, e, _ in turns for t in (s, e) if cs < t < ce})
        bounds = [cs] + cuts + [ce]

        # How many word-level words fall in this chunk's span — used to locate
        # the splits by actual word density rather than by clock time alone.
        lo = bisect.bisect_left(word_starts, cs)
        total = bisect.bisect_left(word_starts, ce) - lo

        prev = 0
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            # A sub-segment inside a real-speech chunk that lands in a diarisation
            # gap falls back to the chunk's dominant speaker rather than being
            # dropped, so no real words are lost at boundary slivers.
            speaker = _speaker_at(turns, (a + b) / 2)
            if speaker == "UNKNOWN":
                speaker = dominant
            if i == len(bounds) - 2:
                end = n  # last sub-segment takes the remaining words
            elif total > 0:
                end = round((bisect.bisect_left(word_starts, b) - lo) / total * n)
            else:
                end = round((b - cs) / (ce - cs) * n)
            end = max(prev, min(end, n))
            sub = words[prev:end]
            prev = end
            if not sub:
                continue
            segments.append({
                "start": a,
                "end": b,
                "text": " ".join(sub),
                "speaker": speaker,
            })
    return segments


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
    # Case-insensitivity is scoped to the trigger phrases and titles via inline
    # (?i:...) flags. The name-capture group stays case-sensitive so that its
    # [A-Z][a-z]+ word boundary actually stops at the end of the name — a global
    # re.IGNORECASE would let lowercase words ("and", "for") match as if
    # capitalised, swallowing them into the detected name.
    name_patterns = [
        r"(?i:my name is|i'm|i am)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",
        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})[,.]?\s+(?i:for the record)",
        r"(?i:senator|chair|secretary|professor|doctor|dr\.?)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})",
    ]
    for segment in merged_segments:
        speaker = segment["speaker"]
        if speaker in speaker_names:
            continue
        for pattern in name_patterns:
            match = re.search(pattern, segment["text"])
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


def _punct_density(segments):
    """Sentence-ending marks per word across all segment text (0 if no words)."""
    text = " ".join(s["text"] for s in segments)
    words = len(text.split())
    marks = sum(text.count(c) for c in ".?!")
    return (marks / words) if words else 1.0


def restore_punctuation(segments, min_density=1 / 60):
    """Repair lowercase/unpunctuated ASR output with an on-device punctuation +
    truecasing model, in place. Skipped if the transcript is already adequately
    punctuated, so it can be applied to a whole batch without degrading the good
    ones. Restores per item to preserve attribution. Returns True if it ran.
    """
    if _punct_density(segments) >= min_density:
        print("      Transcript already punctuated — skipping restoration.")
        return False
    print(f"      Restoring punctuation/casing with {PUNCT_RESTORE_MODEL}...")
    from punctuators.models import PunctCapSegModelONNX  # lazy: optional dependency
    model = PunctCapSegModelONNX.from_pretrained(PUNCT_RESTORE_MODEL)
    texts = [s["text"] for s in segments]
    for seg, out in zip(segments, model.infer(texts)):
        text = " ".join(out) if isinstance(out, list) else out
        # The model emits <unk>/<Unk> for out-of-vocab tokens; strip them
        # (case-insensitively) and collapse whitespace.
        text = re.sub(r"<unk>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            seg["text"] = text
    return True


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

def transcribe(audio_path: str, hf_token: str, word_snap: bool = False,
               restore_punct: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. This script is designed for RunPod GPU pods. "
            "Use transcribe.py for CPU-based local transcription."
        )

    device = "cuda"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    if word_snap:
        print("Word-snap enabled: turn boundaries will snap to speaker changes.")

    input_filename = Path(audio_path).stem
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    # Two caches, one per ASR pass. Delete a cache file to force that pass to
    # re-run. The punctuated pass supplies readable text; the word pass supplies
    # the precise per-word times used to snap turn boundaries (see the hybrid
    # merge below).
    chunk_cache = output_dir / f"{input_filename}_asr_cache.json"   # punctuated text
    word_cache = output_dir / f"{input_filename}_asr_words.json"    # word timestamps

    # ── Step 1: Transcription — two passes, with checkpointing ─────────────────
    # Lazy loaders so a fully-cached run touches neither the model nor the audio.
    _state = {"audio": None, "pipe": None}

    def load_audio():
        if _state["audio"] is None:
            print(f"[1/4] Loading audio file: {audio_path}")
            _state["audio"], _ = librosa.load(audio_path, sr=16000, mono=True)
        return _state["audio"]

    def load_pipe():
        if _state["pipe"] is None:
            print("[1/4] Loading Whisper Large V3 model (float16, sdpa attention)...")
            _state["pipe"] = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=device,
                torch_dtype=torch.float16,
                # Word-level timestamps need cross-attention weights for DTW
                # alignment; flash_attention_2 cannot return them. sdpa is used
                # instead (it falls back to eager for the attention-returning
                # pass), so flash-attn is no longer required by this script.
                model_kwargs={"attn_implementation": "sdpa"},
            )
        return _state["pipe"]

    # Pass A — chunk-level, for punctuated/capitalised text.
    if chunk_cache.exists():
        print("\n[1/4] Found cached punctuated transcript — skipping chunk-level ASR.")
        with open(chunk_cache, "r", encoding="utf-8") as f:
            punct_result = json.load(f)
    else:
        print("\n[1/4] Transcribing for punctuated text (chunk-level pass)...")
        audio = load_audio()
        punct_result = load_pipe()(
            {"raw": audio, "sampling_rate": 16000},
            return_timestamps=True,
            batch_size=24,
            generate_kwargs={"language": "en"},
        )
        with open(chunk_cache, "w", encoding="utf-8") as f:
            json.dump(punct_result, f, ensure_ascii=False, indent=2)

    # Pass B — only for --word-snap: per-word timestamps used to snap boundaries.
    # On a GPU pod large-v3 word-mode is fast enough; on CPU use transcribe.py,
    # which runs this pass on a small model to avoid a large-v3 word-mode stall.
    word_result = None
    if word_snap:
        if word_cache.exists():
            print("[1/4] Found cached word timestamps — skipping word-level ASR.")
            with open(word_cache, "r", encoding="utf-8") as f:
                word_result = json.load(f)
        else:
            print("[1/4] Transcribing for word timestamps (word-level pass)...")
            audio = load_audio()
            # chunk_length_s enables the chunked (batchable) decode path; batch_size
            # then processes several 30 s windows at once on the GPU. This also bounds
            # the memory of the word-level cross-attention alignment.
            word_result = load_pipe()(
                {"raw": audio, "sampling_rate": 16000},
                return_timestamps="word",
                chunk_length_s=30,
                batch_size=24,
                generate_kwargs={"language": "en"},
            )
            with open(word_cache, "w", encoding="utf-8") as f:
                json.dump(word_result, f, ensure_ascii=False, indent=2)

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
    punct_chunks = punct_result.get("chunks", [])
    # Repair punctuation on the chunk text BEFORE merging/snapping, so the model
    # sees sentence-level context (restoring tiny word-snap sub-segments mangles
    # it). No-op if the transcript is already adequately punctuated.
    if restore_punct:
        restore_punctuation(punct_chunks)
    if word_snap:
        print("\n[3/4] Merging punctuated text with speaker labels (word-snapped)...")
        word_chunks = word_result.get("chunks", [])
        merged = merge_punctuated_with_diarization(punct_chunks, word_chunks, diarization)
        print(f"      {len(punct_chunks)} punctuated chunks -> {len(merged)} "
              f"speaker-snapped segments.")
    else:
        print("\n[3/4] Merging transcript with speaker labels...")
        merged = merge_chunks_with_diarization(punct_chunks, diarization)
        print(f"      {len(punct_chunks)} punctuated chunks -> {len(merged)} segments.")

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
    parser.add_argument(
        "--word-snap",
        action="store_true",
        help="Snap turn boundaries to speaker changes using word-level timestamps "
             "(more accurate boundaries, but choppier in rapid exchanges). Adds a "
             "second ASR pass. Default: chunk-level boundaries (smoother).",
    )
    parser.add_argument(
        "--restore-punct",
        action="store_true",
        help="Repair punctuation/capitalisation with an on-device model when "
             "Whisper transcribed the recording lowercase/unpunctuated. Auto-skips "
             "transcripts that are already adequately punctuated.",
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

    transcribe(audio_path, args.hf_token, word_snap=args.word_snap,
               restore_punct=args.restore_punct)
