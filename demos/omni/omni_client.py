#!/usr/bin/env python3
"""
Example client for testing Omni pipeline audio input.
Sends a WAV file and asks what is in the recording.

Usage:
    python3 omni_client.py <path_to_audio.wav>

Requires: openai (pip install openai)
"""

import sys
import base64
from openai import OpenAI

BASE_URL = "http://localhost:11338/v3"
MODEL = "ovms-model"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    audio_format = "mp3" if audio_path.endswith(".mp3") else "wav"

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"Audio: {audio_path} (format: {audio_format})")

    client = OpenAI(base_url=BASE_URL, api_key="unused")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": audio_format,
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is said in this audio recording?",
                    },
                ],
            }
        ],
        max_tokens=256,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
