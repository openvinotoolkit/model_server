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
Voice-to-voice conversational client for Omni pipeline.
Records audio from microphone, sends to server, plays back response.
Maintains conversation history using text transcripts.

Multi turn with audio is not supported yet, @sgonorov

Usage:
    python3 omni_voice_chat.py
    python3 omni_voice_chat.py --voice m02
    python3 omni_voice_chat.py --no-stream
    python3 omni_voice_chat.py --debug

Controls:
    [ENTER]  - Start recording
    [ENTER]  - Stop recording and send
    [q]      - Quit

Requires: pip install sounddevice numpy openai
"""

import argparse
import base64
import copy
import io
import json
import sys
import time
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI

BASE_URL = "http://localhost:11338/v3"
MODEL = "ovms-model"
RECORD_SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 24000
AVAILABLE_VOICES = ["br_f019", "f04", "f245", "f37", "m02", "m31", "m36"]


class AudioRecorder:
    def __init__(self, sample_rate=RECORD_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []

    def start(self):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16",
            callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def stop(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        if not self.frames:
            return None
        return np.concatenate(self.frames, axis=0)

    def to_wav_base64(self, audio):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def redact_payload_for_log(payload):
    """Deep-copy payload and replace base64 blobs with placeholders."""
    out = copy.deepcopy(payload)
    for msg in out.get("input", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "input_audio":
                    ia = part.get("input_audio", {})
                    data = ia.get("data", "")
                    ia["data"] = f"<base64 {len(data)} chars>"
    return out


def stream_response(client, messages, model, voice, audio_format, max_tokens):
    """Send streaming request via OpenAI SDK and yield events."""
    try:
        with client.responses.stream(
            model=model,
            input=messages,
            max_output_tokens=max_tokens,
            extra_body={
                "modalities": ["text", "audio"],
                "audio": {"voice": voice, "format": audio_format},
            },
        ) as stream:
            for event in stream:
                yield event
    except Exception as e:
        print(f"\n  Error: {e}")
        return


def unary_response(client, messages, model, voice, audio_format, max_tokens):
    """Send non-streaming request via OpenAI SDK and return response."""
    try:
        return client.responses.create(
            model=model,
            input=messages,
            max_output_tokens=max_tokens,
            extra_body={
                "modalities": ["text", "audio"],
                "audio": {"voice": voice, "format": audio_format},
            },
        )
    except Exception as e:
        print(f"\n  Error: {e}")
        return None


def play_audio_from_base64(audio_b64, audio_format="wav"):
    """Decode base64 audio and play it."""
    raw = base64.b64decode(audio_b64)
    if audio_format == "wav":
        # WAV file with header — read with wave module
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sample_width == 4:
            # 32-bit float WAV
            samples = np.frombuffer(frames, dtype=np.float32)
            dtype = "float32"
        else:
            # 16-bit int WAV
            samples = np.frombuffer(frames, dtype=np.int16)
            dtype = "int16"
    else:
        # pcm16: raw s16le samples
        samples = np.frombuffer(raw, dtype=np.int16)
        sample_rate = PLAYBACK_SAMPLE_RATE
        dtype = "int16"
    if len(samples) == 0:
        return 0.0
    audio_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype=dtype)
    audio_stream.start()
    audio_stream.write(samples.reshape(-1, 1))
    duration = len(samples) / sample_rate
    time.sleep(duration + 0.1)
    audio_stream.stop()
    audio_stream.close()
    return duration


def main():
    parser = argparse.ArgumentParser(description="Voice-to-voice Omni chat")
    parser.add_argument("--voice", "-v", default="f04", choices=AVAILABLE_VOICES, help="Speaker voice")
    parser.add_argument("--no-stream", action="store_true", help="Use unary (non-streaming) responses")
    parser.add_argument("--save-dir", help="Directory to save audio files for debugging")
    parser.add_argument("--debug", "-d", action="store_true", help="Print debug info (payload, history)")
    args = parser.parse_args()

    streaming = not args.no_stream
    conversation_history = []
    recorder = AudioRecorder()
    turn = 0
    audio_format = "pcm16"  # pcm16 works for both streaming and unary playback
    client = OpenAI(base_url=BASE_URL, api_key="unused")

    print("=" * 50)
    print("  Voice-to-Voice Omni Chat")
    print(f"  Voice: {args.voice} | Mode: {'streaming' if streaming else 'unary'}")
    print("=" * 50)
    print("\n  Press ENTER to start recording.")
    print("  Press ENTER again to stop and send.")
    print("  Type 'q' + ENTER to quit.\n")

    while True:
        user_input = input("  [ENTER to record | q to quit] > ")
        if user_input.strip().lower() == "q":
            print("\n  Goodbye!")
            break

        # Record
        print("  🎤 Recording... (press ENTER to stop)")
        recorder.start()
        input()
        audio = recorder.stop()

        if audio is None or len(audio) == 0:
            print("  No audio recorded, try again.")
            continue

        duration = len(audio) / RECORD_SAMPLE_RATE
        print(f"  Recorded {duration:.1f}s")

        audio_b64 = recorder.to_wav_base64(audio)
        turn += 1

        # Build messages: history + current turn with audio
        messages = list(conversation_history)
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                {"type": "input_text", "text": "(voice message)"}
            ]
        })

        # Debug: log payload
        if args.debug:
            redacted = redact_payload_for_log({"input": messages})
            print(f"\n  [DEBUG] Turn {turn} input ({len(messages)} messages):")
            print(f"  {json.dumps(redacted['input'], indent=4)}")

        response_text = ""
        audio_data_b64 = ""
        send_time = time.time()

        if streaming:
            # --- Streaming path ---
            print(f"\n  Assistant: ", end="", flush=True)

            audio_stream = sd.OutputStream(samplerate=PLAYBACK_SAMPLE_RATE, channels=1, dtype="int16")
            audio_stream.start()
            audio_chunks = []
            audio_playing = False
            first_text_time = None
            first_audio_time = None

            for event in stream_response(client, messages, MODEL, args.voice, audio_format, 512):
                event_type = event.type

                if event_type == "response.output_text.delta":
                    delta = event.delta
                    response_text += delta
                    print(delta, end="", flush=True)
                    if first_text_time is None:
                        first_text_time = time.time()

                elif event_type == "response.audio.delta":
                    if not audio_playing:
                        audio_playing = True
                        first_audio_time = time.time()
                        text_latency = first_audio_time - send_time
                        print(f"\n  🔊 Playing... (text took {text_latency:.1f}s)", flush=True)
                    raw = base64.b64decode(event.delta)
                    samples = np.frombuffer(raw, dtype=np.int16)
                    audio_stream.write(samples.reshape(-1, 1))
                    audio_chunks.append(samples)

            audio_stream.stop()
            audio_stream.close()

            total_audio_samples = sum(len(c) for c in audio_chunks)
            total_elapsed = time.time() - send_time
            if total_audio_samples > 0:
                audio_dur = total_audio_samples / PLAYBACK_SAMPLE_RATE
                if first_audio_time:
                    wall_time = time.time() - first_audio_time
                    rtf = wall_time / audio_dur if audio_dur > 0 else 0
                    print(f"\n  [{audio_dur:.1f}s audio, RTF: {rtf:.1f}x, total: {total_elapsed:.1f}s]")
                else:
                    print(f"\n  [{audio_dur:.1f}s audio, total: {total_elapsed:.1f}s]")
            else:
                print(f"\n  [text only, {total_elapsed:.1f}s]")

        else:
            # --- Unary path ---
            print(f"\n  Waiting for response...", flush=True)
            result = unary_response(client, messages, MODEL, args.voice, audio_format, 512)
            if result is None:
                continue

            elapsed = time.time() - send_time

            if args.debug:
                print(f"\n  [DEBUG] Response id: {result.id}, status: {result.status}")

            # Extract text and audio from response
            audio_data_b64 = ""
            for item in result.output:
                if item.type == "message":
                    for part in item.content:
                        if part.type == "output_text":
                            response_text = part.text
                        elif part.type == "output_audio":
                            audio_data_b64 = part.data if hasattr(part, 'data') else ""

            print(f"  Assistant: {response_text}")

            if audio_data_b64:
                print(f"  🔊 Playing... (generated in {elapsed:.1f}s)")
                audio_dur = play_audio_from_base64(audio_data_b64, audio_format)
                print(f"  [{audio_dur:.1f}s audio]")
            else:
                print(f"  [no audio, {elapsed:.1f}s]")

        print()

        # Append to conversation history
        # Keep audio in history so the model can reference previous voice turns
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ]
        })
        if response_text:
            conversation_history.append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": response_text}]
            })

        if args.debug:
            redacted_history = []
            for msg in conversation_history:
                redacted_history.append(redact_payload_for_log({"input": [msg]})["input"][0])
            print(f"  [DEBUG] Conversation history ({len(conversation_history)} messages):")
            print(f"  {json.dumps(redacted_history, indent=4)}\n")

        # Save debug audio
        if args.save_dir:
            import os
            os.makedirs(args.save_dir, exist_ok=True)
            in_path = os.path.join(args.save_dir, f"turn{turn:02d}_input.wav")
            with wave.open(in_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RECORD_SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

        print(f"  [Turn {turn} | History: {len(conversation_history)} messages]")


if __name__ == "__main__":
    main()
