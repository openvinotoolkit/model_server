#!/usr/bin/env python3
"""
Example client for testing Omni pipeline with audio OUTPUT.
Sends a text prompt and requests audio+text response.
Saves the generated speech to a WAV file.

Usage:
    python3 omni_audio_output_client.py [--output output.wav]

Requires: openai (pip install openai)
"""

import sys
import base64
import argparse
from openai import OpenAI

BASE_URL = "http://localhost:11338/v3"
MODEL = "ovms-model"


def main():
    parser = argparse.ArgumentParser(description="Test Omni audio output")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file path")
    parser.add_argument("--prompt", "-p", default="Hello, how are you today?", help="Text prompt")
    args = parser.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="unused")

    response = client.chat.completions.create(
        model=MODEL,
        modalities=["text", "audio"],
        audio={"voice": "default", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": args.prompt,
            }
        ],
        max_tokens=256,
    )

    message = response.choices[0].message
    print(f"Text: {message.content}")

    if hasattr(message, "audio") and message.audio:
        audio_data = base64.b64decode(message.audio.data)
        with open(args.output, "wb") as f:
            f.write(audio_data)
        print(f"Audio saved to: {args.output} ({len(audio_data)} bytes)")
        if hasattr(message.audio, "transcript"):
            print(f"Transcript: {message.audio.transcript}")
    else:
        print("No audio in response")


if __name__ == "__main__":
    main()
