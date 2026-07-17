#!/usr/bin/env python3
"""
Streaming audio client using OpenAI Python SDK (Responses API).
This validates that our Responses API streaming implementation is fully
compatible with the official OpenAI client library.

T = Text Input
AT = Audio & Text Output

Usage:
    python3 omni_responses_stream_T_AT_openai.py --prompt "Tell me a short story"
    python3 omni_responses_stream_T_AT_openai.py --prompt "Hello" --voice m02

Requires: pip install openai sounddevice numpy
"""

import argparse
import base64
import time

import numpy as np
import sounddevice as sd
from openai import OpenAI

BASE_URL = "http://localhost:11338/v3"
MODEL = "ovms-model"
SAMPLE_RATE = 24000
AVAILABLE_VOICES = ["br_f019", "f04", "f245", "f37", "m02", "m31", "m36"]


def main():
    parser = argparse.ArgumentParser(description="Stream audio via OpenAI SDK (Responses API)")
    parser.add_argument("--prompt", "-p", default="Hello, how are you today?", help="Text prompt")
    parser.add_argument("--voice", "-v", default="f04", choices=AVAILABLE_VOICES, help="Speaker voice")
    parser.add_argument("--save", "-s", help="Optional: save audio to WAV file")
    args = parser.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="unused")

    print(f"Prompt: {args.prompt}")
    print(f"Voice: {args.voice}")
    print(f"Using OpenAI SDK → {BASE_URL}/responses (streaming)\n")
    print("--- Text ---")

    audio_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    audio_stream.start()

    all_audio_chunks = []
    audio_started = False
    first_audio_time = None
    total_audio_samples = 0

    with client.responses.stream(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": args.prompt}],
            }
        ],
        max_output_tokens=256,
        extra_body={
            "modalities": ["text", "audio"],
            "audio": {"voice": args.voice, "format": "pcm16"},
        },
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

            elif event.type == "response.output_text.done":
                print("\n\n--- Audio streaming ---")

            elif event.type == "response.audio.delta":
                now = time.time()
                if not audio_started:
                    audio_started = True
                    first_audio_time = now
                    print("Playing audio...", flush=True)
                raw = base64.b64decode(event.delta)
                samples = np.frombuffer(raw, dtype=np.int16)
                total_audio_samples += len(samples)
                audio_stream.write(samples.reshape(-1, 1))
                all_audio_chunks.append(samples)

            elif event.type == "response.audio.done":
                print("Audio stream complete.")

    audio_stream.stop()
    audio_stream.close()

    # Print real-time factor
    if first_audio_time and total_audio_samples > 0:
        wall_clock_s = time.time() - first_audio_time
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
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(all_samples.tobytes())
        print(f"Saved to: {args.save}")


if __name__ == "__main__":
    main()
