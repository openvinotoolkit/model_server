#!/usr/bin/env python3
#
# Copyright (c) 2024 Intel Corporation
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
Omni model client — Chat Completions API.

Supports any combination of text, audio, and image input.
Can request text-only or text+audio output.

Examples:
    # Text only (text response)
    python3 chat_completions.py --prompt "What is OpenVINO?"

    # Text input, audio+text output
    python3 chat_completions.py --prompt "Say hello" --audio-output --voice f04

    # Audio input (text response)
    python3 chat_completions.py --audio recording.wav

    # Audio input, audio+text output
    python3 chat_completions.py --audio recording.wav --audio-output

    # Image + text input
    python3 chat_completions.py --image photo.jpg --prompt "Describe this"

    # All modalities: audio + image + text → audio + text
    python3 chat_completions.py --audio recording.wav --image photo.jpg --prompt "Describe both" --audio-output --voice m02

    # Save audio output to file
    python3 chat_completions.py --prompt "Hello" --audio-output --save output.wav

Requires: pip install openai
"""

import argparse
import base64
from openai import OpenAI

VOICES = ["br_f019", "f04", "f245", "f37", "m02", "m31", "m36"]


def main():
    parser = argparse.ArgumentParser(description="Omni Chat Completions client")
    parser.add_argument("--prompt", "-p", default="Hello, how are you?", help="Text prompt")
    parser.add_argument("--audio", "-a", help="Input audio file (WAV or MP3)")
    parser.add_argument("--image", "-i", help="Input image file (JPG or PNG)")
    parser.add_argument("--audio-output", action="store_true", help="Request audio in response")
    parser.add_argument("--voice", "-v", default="f04", choices=VOICES, help="Voice for audio output")
    parser.add_argument("--save", "-s", help="Save audio output to WAV file")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--url", default="http://localhost:11338/v3", help="Server base URL")
    parser.add_argument("--model", default="ovms-model", help="Model name")
    args = parser.parse_args()

    # Build content parts
    content = []

    if args.audio:
        fmt = "mp3" if args.audio.endswith(".mp3") else "wav"
        with open(args.audio, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}})
        print(f"Audio input: {args.audio}")

    if args.image:
        mime = "image/png" if args.image.endswith(".png") else "image/jpeg"
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}})
        print(f"Image input: {args.image}")

    content.append({"type": "text", "text": args.prompt})

    # Build request
    kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
    }

    if args.audio_output:
        kwargs["modalities"] = ["text", "audio"]
        kwargs["audio"] = {"voice": args.voice, "format": "wav"}
        print(f"Voice: {args.voice}")

    # Send request
    client = OpenAI(base_url=args.url, api_key="unused")
    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message

    # Print text
    print(f"\nResponse: {message.content}")

    # Handle audio output
    if hasattr(message, "audio") and message.audio:
        audio_bytes = base64.b64decode(message.audio.data)
        print(f"Audio: {len(audio_bytes)} bytes")
        if message.audio.transcript:
            print(f"Transcript: {message.audio.transcript}")
        if args.save:
            with open(args.save, "wb") as f:
                f.write(audio_bytes)
            print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
