#!/usr/bin/env python3
"""
Streaming audio playback client for Omni pipeline.
Sends a text prompt, streams text tokens to console,
then streams audio chunks and plays them in real-time.

T = Text Input
AT = Audio & Text Output

Usage:
    python3 omni_responses_stream_T_AT.py --prompt "Tell me a short story"
    python3 omni_responses_stream_T_AT.py --prompt "Hello" --voice m02

Requires: pip install sounddevice numpy requests
"""

import argparse
import base64
import json
import sys

import numpy as np
import sounddevice as sd
import requests

BASE_URL = "http://localhost:11338/v3"
MODEL = "ovms-model"
SAMPLE_RATE = 24000
AVAILABLE_VOICES = ["br_f019", "f04", "f245", "f37", "m02", "m31", "m36"]


def main():
    parser = argparse.ArgumentParser(description="Stream audio from Omni model and play in real-time")
    parser.add_argument("--prompt", "-p", default="Hello, how are you today?", help="Text prompt")
    parser.add_argument("--voice", "-v", default="f04", choices=AVAILABLE_VOICES, help="Speaker voice")
    parser.add_argument("--save", "-s", help="Optional: save audio to WAV file")
    args = parser.parse_args()

    payload = {
        "model": MODEL,
        "stream": True,
        "modalities": ["text", "audio"],
        "audio": {"voice": args.voice, "format": "pcm16"},
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": args.prompt}
                ]
            }
        ],
        "max_output_tokens": 256,
    }

    print(f"Prompt: {args.prompt}")
    print(f"Voice: {args.voice}")
    print(f"Streaming from {BASE_URL}/responses ...\n")
    print("--- Text ---")

    resp = requests.post(f"{BASE_URL}/responses", json=payload, stream=True, timeout=120)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    # Open audio output stream
    audio_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    audio_stream.start()

    all_audio_chunks = []
    audio_started = False
    first_audio_time = None
    last_audio_time = None
    total_audio_samples = 0

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "response.output_text.delta":
            # Print text as it arrives
            print(event.get("delta", ""), end="", flush=True)

        elif event_type == "response.output_text.done":
            print("\n\n--- Audio streaming ---")

        elif event_type == "response.audio.delta":
            import time
            now = time.time()
            if not audio_started:
                audio_started = True
                first_audio_time = now
                print("Playing audio...", flush=True)
            last_audio_time = now
            # Decode base64 pcm16 chunk and play
            raw = base64.b64decode(event["delta"])
            samples = np.frombuffer(raw, dtype=np.int16)
            total_audio_samples += len(samples)
            audio_stream.write(samples.reshape(-1, 1))
            all_audio_chunks.append(samples)

        elif event_type == "response.audio.done":
            print("Audio stream complete.")

    audio_stream.stop()
    audio_stream.close()

    # Print real-time factor
    if first_audio_time and last_audio_time and total_audio_samples > 0:
        wall_clock_s = last_audio_time - first_audio_time
        audio_duration_s = total_audio_samples / SAMPLE_RATE
        if wall_clock_s > 0:
            rtf = wall_clock_s / audio_duration_s
            print(f"\n--- Performance ---")
            print(f"Audio duration: {audio_duration_s:.2f} s ({total_audio_samples} samples)")
            print(f"Wall-clock time: {wall_clock_s:.2f} s")
            print(f"Real-time factor: {rtf:.2f}x (1.0 = real-time, lower = faster)")

    # Optionally save to file
    if args.save and all_audio_chunks:
        import wave
        all_samples = np.concatenate(all_audio_chunks)
        with wave.open(args.save, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(all_samples.tobytes())
        print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
