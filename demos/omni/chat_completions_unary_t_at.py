#!/usr/bin/env python3
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Example client for testing Omni pipeline with audio OUTPUT.
Sends a text prompt and requests audio+text response.
Saves the generated speech to a WAV file.

T = Text Input
AT = Audio & Text Output

Usage:
    python3 omni_ccompletion_unary_T_AT.py [--output output.wav]

Requires: openai (pip install openai)
"""

import sys
import base64
import argparse
from openai import OpenAI

BASE_URL = "http://localhost:11338/v3"
MODEL = "ovms-model"


AVAILABLE_VOICES = ["br_f019", "f04", "f245", "f37", "m02", "m31", "m36"]


def main():
    parser = argparse.ArgumentParser(description="Test Omni audio output")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file path")
    parser.add_argument("--prompt", "-p", default="Hello, how are you today?", help="Text prompt")
    parser.add_argument("--voice", "-v", default="f04", choices=AVAILABLE_VOICES,
                        help=f"Speaker voice ({', '.join(AVAILABLE_VOICES)})")
    args = parser.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="unused")

    response = client.chat.completions.create(
        model=MODEL,
        modalities=["text", "audio"],
        audio={"voice": args.voice, "format": "wav"},
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
