#!/usr/bin/env python3
"""
Example client for testing Omni pipeline audio input via Responses API.
Sends a WAV audio file and asks what is in the recording.

AT = Audio & Text Input
T = Text Output

Usage:
    python3 omni_responses_unary_AT_T.py <path_to_audio.wav>

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

    response = client.responses.create(
        model=MODEL,
        input=[
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
                        "type": "input_text",
                        "text": "What is said in this audio recording?",
                    },
                ],
            }
        ],
        max_output_tokens=256,
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
