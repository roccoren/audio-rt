import argparse
import asyncio
import sys
from typing import Optional, Sequence

import numpy as np

from voice_client import (
    AzureVoiceClient,
    float_to_pcm16,
    generate_and_save,
    load_env_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talk to the Zara persona via Azure OpenAI Voice Live and save the audio reply."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Optional text prompt to send to Zara. If omitted, microphone audio will be used.",
    )
    parser.add_argument(
        "--microphone",
        action="store_true",
        help="Capture microphone audio instead of sending text.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Microphone capture length in seconds when --microphone is set. Default is 8.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Override the default Azure OpenAI voice name.",
    )
    return parser.parse_args()


def capture_microphone(duration: float, sample_rate: int) -> Sequence[bytes]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required for microphone capture. Install optional dependencies."
        ) from exc

    print(f"Recording for {duration:.1f} seconds...", file=sys.stderr)
    frames = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    mono = frames.reshape(-1)
    pcm_bytes = float_to_pcm16(mono.astype(np.float32))
    return [pcm_bytes]


def main() -> None:
    args = parse_args()

    if not args.text and not args.microphone:
        print("Provide --text or use --microphone to capture audio.", file=sys.stderr)
        sys.exit(1)

    config = load_env_config()
    client = AzureVoiceClient(config)

    audio_chunks: Optional[Sequence[bytes]] = None
    if args.microphone:
        audio_chunks = capture_microphone(args.duration, config.sample_rate_hz)

    transcript, audio_path = asyncio.run(
        generate_and_save(
            client,
            text_prompt=args.text,
            audio_chunks=audio_chunks,
            voice=args.voice,
        )
    )

    print("Zara said:")
    print(transcript)
    print(f"Audio saved to {audio_path}")


if __name__ == "__main__":
    main()
