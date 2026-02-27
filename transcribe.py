import torch
import argparse
from transformers import pipeline
from pathlib import Path

def transcribe(audio_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading model — this may take a minute on first run while the model downloads...")

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )

    print(f"Transcribing: {audio_path}")
    result = pipe(audio_path, return_timestamps=False)

    # Save output
    input_filename = Path(audio_path).stem
    output_path = Path("output") / f"{input_filename}.txt"
    output_path.write_text(result["text"], encoding="utf-8")

    print(f"\nTranscription complete. Saved to: {output_path}")
    print("\n--- Preview (first 500 characters) ---")
    print(result["text"][:500])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("audio", help="Path to the audio file (e.g. audio/recording.mp3)")
    args = parser.parse_args()
    transcribe(args.audio)