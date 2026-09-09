#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Verify that both the HuggingFace and the OpenVINO versions of the dummy LLM
generate the expected sequence, ending in a valid end-of-sequence token.

Expected pattern (ignoring the echoed prompt), with normal (EOS-respecting)
generation settings:
    OpenVINO is an open-source toolkit created by Intel to speed up and run AI models efficiently.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HF_DIR = os.path.join(SCRIPT_DIR, "hf_model")
DEFAULT_OV_DIR = os.path.join(SCRIPT_DIR, "ov_model")

DEFAULT_EXPECTED_FRAGMENT = "OpenVINO is an open-source toolkit"
MAX_NEW_TOKENS = 60


def load_metadata(hf_dir: str) -> dict:
    """Return metadata written by create_model.py, falling back to defaults."""
    path = os.path.join(hf_dir, "dummy_llm_metadata.json")
    if os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def verify_hf(model_dir: str, expected: str, max_new_tokens: int) -> bool:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"\n=== HuggingFace model: {model_dir} ===")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens
    n_prompt = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output_ids[0][n_prompt:], skip_special_tokens=True)

    print(f"Prompt   : {prompt!r}")
    print(f"Generated: {generated!r}")

    ok = expected in generated
    print(f"Result   : {'PASS' if ok else 'FAIL'} (expected fragment {expected!r})")
    return ok


def verify_ov_llm_pipeline(model_dir: str, expected: str, max_new_tokens: int) -> bool:
    """Test with openvino_genai.LLMPipeline (equivalent to OVMS pipeline type LM)."""
    try:
        import openvino_genai as ov_genai
    except ImportError:
        print("openvino_genai not installed – skipping LLMPipeline test.")
        return True

    print(f"\n=== OpenVINO LLMPipeline: {model_dir} ===")
    try:
        pipe = ov_genai.LLMPipeline(model_dir, "CPU")
        result = pipe.generate("Hello", max_new_tokens=max_new_tokens, do_sample=False)
        print(f"Generated: {result!r}")
        ok = expected in result
        print(f"Result   : {'PASS' if ok else 'FAIL'} (expected fragment {expected!r})")
        return ok
    except Exception as exc:
        print(f"FAIL: LLMPipeline raised {exc}")
        return False


def verify_ov_cb_pipeline(model_dir: str, expected: str, max_new_tokens: int) -> bool:
    """Test with openvino_genai.ContinuousBatchingPipeline (OVMS pipeline type LM_CB)."""
    try:
        import openvino_genai as ov_genai
    except ImportError:
        print("openvino_genai not installed – skipping ContinuousBatchingPipeline test.")
        return True

    print(f"\n=== OpenVINO ContinuousBatchingPipeline: {model_dir} ===")
    try:
        scheduler_cfg = ov_genai.SchedulerConfig()
        scheduler_cfg.max_num_batched_tokens = 256
        scheduler_cfg.cache_size = 1

        pipe = ov_genai.ContinuousBatchingPipeline(model_dir, scheduler_cfg, "CPU")
        outputs = pipe.generate(
            ["Hello"],
            [ov_genai.GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)],
        )
        result = outputs[0].m_generation_ids[0]
        print(f"Generated: {result!r}")
        ok = expected in result
        print(f"Result   : {'PASS' if ok else 'FAIL'} (expected fragment {expected!r})")
        return ok
    except Exception as exc:
        print(f"FAIL: ContinuousBatchingPipeline raised {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dummy LLM models")
    parser.add_argument("--hf-model-dir", default=DEFAULT_HF_DIR)
    parser.add_argument("--ov-model-dir",  default=DEFAULT_OV_DIR)
    parser.add_argument("--skip-hf",  action="store_true", help="Skip HuggingFace model test")
    parser.add_argument("--skip-ov",  action="store_true", help="Skip OpenVINO model tests")
    args = parser.parse_args()

    results: list[bool] = []

    meta = load_metadata(args.hf_model_dir)
    # Use the first token of the sequence as the expected fragment (always present in output)
    expected = meta.get("token_texts", [""])[0] or DEFAULT_EXPECTED_FRAGMENT
    # Use enough tokens to cover at least one full pass
    n_seq = len(meta.get("token_ids", [1]))
    max_new_tokens = max(MAX_NEW_TOKENS, n_seq + 5)

    if not args.skip_hf:
        if os.path.isdir(args.hf_model_dir):
            results.append(verify_hf(args.hf_model_dir, expected, max_new_tokens))
        else:
            print(f"HF model dir not found: {args.hf_model_dir} – skipping.")

    if not args.skip_ov:
        if os.path.isdir(args.ov_model_dir):
            results.append(verify_ov_llm_pipeline(args.ov_model_dir, expected, max_new_tokens))
            results.append(verify_ov_cb_pipeline(args.ov_model_dir, expected, max_new_tokens))
        else:
            print(f"OV model dir not found: {args.ov_model_dir} – skipping.")

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*50}")
    print(f"Summary: {passed}/{total} checks passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
