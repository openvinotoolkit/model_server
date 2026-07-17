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
Full multimodal Omni client: audio + image + text input → text + audio output.

ATI = Audio, Text & Image Input
AT = Audio & Text Output

Usage:
    python3 omni_ccompletion_unary_ATI_AT.py --audio recording.wav --image photo.jpg
    python3 omni_ccompletion_unary_ATI_AT.py --audio recording.wav --prompt "Transcribe this"
    python3 omni_ccompletion_unary_ATI_AT.py --image photo.jpg --prompt "Describe this image"
    python3 omni_ccompletion_unary_ATI_AT.py --audio recording.wav --image photo.jpg --voice m02

Requires: openai (pip install openai)
"""

import base64
import argparse
from openai import OpenAI

BASE_URL = "http://localhost:11338/v3"
MODEL = "ovms-model"
AVAILABLE_VOICES = ["br_f019", "f04", "f245", "f37", "m02", "m31", "m36"]


def main():
    parser = argparse.ArgumentParser(description="Full multimodal Omni client")
    parser.add_argument("--audio", "-a", help="Input audio file (wav or mp3)")
    parser.add_argument("--image", "-i", help="Input image file (jpg or png)")
    parser.add_argument("--prompt", "-p", default="Describe what you see and hear.",
                        help="Text prompt")
    parser.add_argument("--voice", "-v", default="f04", choices=AVAILABLE_VOICES,
                        help="Speaker voice for audio output")
    parser.add_argument("--output", "-o", default="output.wav",
                        help="Output WAV file path")
    parser.add_argument("--no-audio-output", action="store_true",
                        help="Disable audio output (text only)")
    args = parser.parse_args()

    if not args.audio and not args.image:
        parser.error("Provide at least --audio or --image")

    content = []

    if args.audio:
        audio_format = "mp3" if args.audio.endswith(".mp3") else "wav"
        with open(args.audio, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": audio_format},
        })
        print(f"Audio input: {args.audio} (format: {audio_format})")

    if args.image:
        mime = "image/png" if args.image.endswith(".png") else "image/jpeg"
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
        })
        print(f"Image input: {args.image}")

    content.append({"type": "text", "text": args.prompt})

    client = OpenAI(base_url=BASE_URL, api_key="unused")

    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
    }

    if not args.no_audio_output:
        kwargs["modalities"] = ["text", "audio"]
        kwargs["audio"] = {"voice": args.voice, "format": "wav"}

    response = client.chat.completions.create(**kwargs)

    message = response.choices[0].message
    print(f"\nText: {message.content}")

    if hasattr(message, "audio") and message.audio:
        audio_data = base64.b64decode(message.audio.data)
        with open(args.output, "wb") as f:
            f.write(audio_data)
        print(f"Audio saved to: {args.output} ({len(audio_data)} bytes)")


if __name__ == "__main__":
    main()
