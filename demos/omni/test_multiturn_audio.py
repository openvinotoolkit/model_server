#!/usr/bin/env python3
"""Check multi-turn Omni audio placement without a WebRTC connection.

Run from the repository root: python3 demos/omni/test_multiturn_audio.py
Requires host ffmpeg and a build container with the Qwen3-Omni model mounted.
"""

import argparse
import struct
import subprocess
import sys
from pathlib import Path


def decode_webm(path: Path) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        raise ValueError(f"No audio decoded from {path}")
    return result.stdout


def run_in_container(model_path: str) -> None:
    import numpy as np
    import openvino_genai as genai
    from openvino import Tensor

    sizes = sys.stdin.buffer.read(8)
    if len(sizes) != 8:
        raise ValueError("Expected two audio lengths on stdin")
    first_size, second_size = struct.unpack("<II", sizes)
    audio_data = sys.stdin.buffer.read()
    if len(audio_data) != first_size + second_size:
        raise ValueError("Incomplete audio data on stdin")
    first_audio = Tensor(np.frombuffer(audio_data[:first_size], dtype=np.float32).copy())
    second_audio = Tensor(np.frombuffer(audio_data[first_size:], dtype=np.float32).copy())

    tokenizer = genai.Tokenizer(model_path)
    pipeline = genai.OmniPipeline(model_path, "CPU")
    text_config = genai.GenerationConfig()
    text_config.max_new_tokens = 64
    text_config.do_sample = False
    text_config.apply_chat_template = False
    speech_config = genai.OmniTalkerSpeechConfig(model_path)
    speech_config.return_audio = False

    history = [{"role": "system", "content": "Respond in English only. Keep answers short."}]
    audios = []
    for audio, expected in ((first_audio, "elephant"), (second_audio, "parrot")):
        audio_index = len(audios)
        audios.append(audio)
        history.append({"role": "user", "content": f"Audio input <ov_genai_audio_{audio_index}>"})
        prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)
        result = pipeline.generate(
            prompt, audios=audios, text_config=text_config, talker_speech_config=speech_config
        )
        answer = result.texts[0].strip()
        print(f"Turn {audio_index + 1}: {answer}", flush=True)
        normalized_answer = answer.casefold()
        if expected not in normalized_answer:
            raise AssertionError(f"Turn {audio_index + 1}: expected {expected!r}, got {answer!r}")
        if audio_index == 0 and "parrot" in normalized_answer:
            raise AssertionError(f"Turn 1 leaked the second answer: {answer!r}")
        history.append({"role": "assistant", "content": answer})
    print("Both audio turns passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="ovms-build-krz")
    parser.add_argument("--model", default="/ovms/models/OpenVINO/qwen3_omni_dense_int4")
    parser.add_argument("--first", type=Path, default=Path(__file__).resolve().parent / "1SayLargestAnimal.webm")
    parser.add_argument("--second", type=Path, default=Path(__file__).resolve().parent / "2NowSaySmallestAnimal.webm")
    parser.add_argument("--in-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.in_container:
        run_in_container(args.model)
        return

    first_audio = decode_webm(args.first)
    second_audio = decode_webm(args.second)
    payload = struct.pack("<II", len(first_audio), len(second_audio)) + first_audio + second_audio
    subprocess.run(
        ["docker", "exec", "-i", args.container, "python3", "/ovms/demos/omni/test_multiturn_audio.py",
         "--in-container", "--model", args.model],
        input=payload,
        check=True,
    )


if __name__ == "__main__":
    main()