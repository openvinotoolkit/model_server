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
Omni model client — Responses API.

Supports any combination of text, audio, and image input.
Can request text-only or text+audio output, with optional streaming.

Examples:
    # Text only (text response)
    python3 responses.py --prompt "What is OpenVINO?"

    # Text input, audio+text output
    python3 responses.py --prompt "Say hello" --audio-output --voice f04

    # Streaming text + audio
    python3 responses.py --prompt "Tell me a story" --audio-output --stream

    # Audio input (text response)
    python3 responses.py --audio recording.wav

    # Audio input, streaming audio+text output
    python3 responses.py --audio recording.wav --audio-output --stream

    # Image + text input
    python3 responses.py --image photo.jpg --prompt "Describe this"

    # All modalities with streaming
    python3 responses.py --audio recording.wav --image photo.jpg --prompt "Describe both" --audio-output --stream --voice m02

    # Save streamed audio to file
    python3 responses.py --prompt "Hello" --audio-output --stream --save output.wav

Requires: pip install openai numpy sounddevice (sounddevice only needed for --stream audio playback)
"""

import argparse
import base64
from openai import OpenAI

VOICES = ["br_f019", "f04", "f245", "f37", "m02", "m31", "m36"]


def build_content(args):
    """Build content parts from arguments."""
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
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{image_b64}"})
        print(f"Image input: {args.image}")

    content.append({"type": "input_text", "text": args.prompt})

    return content


def run_unary(client, args, content):
    """Send a non-streaming request and print the response."""
    kwargs = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": args.max_tokens,
    }

    if args.audio_output:
        # This is not OpenAI standard
        kwargs["extra_body"] = {
            "modalities": ["text", "audio"] if not args.audio_only_output else ["audio"],
            "audio": {"voice": args.voice, "format": "wav"},
            "chunk_frames": args.chunk_frames,
        }

    response = client.responses.create(**kwargs)

    # Extract text and audio
    text = ""
    audio_b64 = ""
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    text = part.text
                elif part.type == "output_audio":
                    audio_b64 = part.data if hasattr(part, "data") else ""

    if text:
        print(f"\nResponse: {text}")

    if audio_b64 and args.save:
        audio_bytes = base64.b64decode(audio_b64)
        with open(args.save, "wb") as f:
            f.write(audio_bytes)
        print(f"Audio saved to: {args.save} ({len(audio_bytes)} bytes)")


def run_streaming(client, args, content):
    """Send a streaming request, print text, and play audio in real-time."""
    import time
    import threading
    import collections
    import numpy as np
    import sounddevice as sd

    SAMPLE_RATE = 24000

    kwargs = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": args.max_tokens,
    }

    if args.audio_output:
        kwargs["extra_body"] = {
            "modalities": ["text", "audio"] if not args.audio_only_output else ["audio"],
            "audio": {"voice": args.voice, "format": "pcm16"},
            "chunk_frames": args.chunk_frames,
        }

    # Non-blocking audio playback using a ring buffer
    import collections
    audio_buffer = collections.deque()  # deque of int16 samples (flat)
    buffer_lock = threading.Lock()

    def audio_callback(outdata, frames, time_info, status):
        """Called by sounddevice when it needs more samples."""
        with buffer_lock:
            available = len(audio_buffer)
            n = min(frames, available)
            for i in range(n):
                outdata[i, 0] = audio_buffer.popleft()
            # Fill remainder with silence
            if n < frames:
                outdata[n:, 0] = 0

    audio_stream = None
    if args.audio_output:
        audio_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=audio_callback, blocksize=1024)
        audio_stream.start()

    all_audio = []
    audio_started = False
    start_time = time.time()
    last_chunk_time = None

    print()
    with client.responses.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

            elif event.type == "response.output_text.done":
                if args.audio_output:
                    print("\n\n[Audio streaming...]")

            elif event.type == "response.audio.delta":
                now = time.time()
                if last_chunk_time is not None:
                    print(f"  inter-chunk: {(now - last_chunk_time)*1000:.1f} ms", end="")

                if not audio_started:
                    audio_started = True
                    print(f"  [first chunk at {(now - start_time)*1000:.0f} ms]")

                decode_start = time.time()
                raw = base64.b64decode(event.delta)
                samples = np.frombuffer(raw, dtype=np.int16)
                decode_ms = (time.time() - decode_start) * 1000

                with buffer_lock:
                    audio_buffer.extend(samples)
                all_audio.append(samples)

                audio_ms = len(samples) / SAMPLE_RATE * 1000
                print(f"  decode: {decode_ms:.2f} ms, samples: {len(samples)}, audio: {audio_ms:.0f} ms")
                last_chunk_time = time.time()

            elif event.type == "response.audio.done":
                pass

    # Wait for playback to finish
    if audio_stream and all_audio:
        # Wait until buffer is drained (audio finishes playing)
        while True:
            with buffer_lock:
                if len(audio_buffer) == 0:
                    break
            time.sleep(0.05)
        # Extra buffer for the sounddevice internal buffer to flush
        time.sleep(0.2)

    if audio_stream:
        audio_stream.stop()
        audio_stream.close()

    # Stats
    total_samples = sum(len(c) for c in all_audio)
    elapsed = time.time() - start_time
    if total_samples > 0:
        duration = total_samples / SAMPLE_RATE
        print(f"\n[Audio: {duration:.1f}s, total time: {elapsed:.1f}s, RTF: {elapsed / duration:.1f}x]")
    else:
        print(f"\n[Time: {elapsed:.1f}s]")

    # Save
    if args.save and all_audio:
        import wave
        merged = np.concatenate(all_audio)
        with wave.open(args.save, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(merged.tobytes())
        print(f"Saved to: {args.save}")


def main():
    parser = argparse.ArgumentParser(description="Omni Responses API client")
    parser.add_argument("--prompt", "-p", default="Hello, how are you?", help="Text prompt")
    parser.add_argument("--audio", "-a", help="Input audio file (WAV or MP3)")
    parser.add_argument("--image", "-i", help="Input image file (JPG or PNG)")
    parser.add_argument("--audio-output", action="store_true", help="Request audio in response")
    parser.add_argument("--audio-only-output", action="store_true", help="Request ONLY audio in response")
    parser.add_argument("--voice", "-v", default="f04", choices=VOICES, help="Voice for audio output")
    parser.add_argument("--chunk-frames", type=int, default=4, help="Audio chunk frames (each frame=80ms, default=4=320ms chunks)")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument("--save", "-s", help="Save audio output to file")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--url", default="http://localhost:11338/v3", help="Server base URL")
    parser.add_argument("--model", default="ovms-model", help="Model name")
    args = parser.parse_args()

    if args.audio_only_output:
        assert args.audio_output

    if args.audio_output:
        print(f"Voice: {args.voice}")

    content = build_content(args)
    client = OpenAI(base_url=args.url, api_key="unused")

    if args.stream:
        run_streaming(client, args, content)
    else:
        run_unary(client, args, content)


if __name__ == "__main__":
    main()
