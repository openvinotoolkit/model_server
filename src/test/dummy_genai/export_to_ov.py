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
Export the dummy HuggingFace model to OpenVINO IR.

Two steps:
  1. optimum-cli export openvino  →  openvino_model.xml/.bin  (stateful, KV-cache as state)
  2. convert_tokenizer             →  openvino_tokenizer.xml/.bin + openvino_detokenizer.xml/.bin

The resulting directory is loadable by:
  - openvino_genai.LLMPipeline                    (OVMS pipeline type LM)
  - openvino_genai.ContinuousBatchingPipeline     (OVMS pipeline type LM_CB)
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HF_DIR  = os.path.join(SCRIPT_DIR, "hf_model")
DEFAULT_OV_DIR  = os.path.join(SCRIPT_DIR, "ov_model")


def run(cmd: str) -> bool:
    print(f"  $ {cmd}")
    return os.system(cmd) == 0


def export_model(hf_dir: str, ov_dir: str, weight_format: str) -> None:
    """Run optimum-cli to convert the HF model to OpenVINO IR."""
    if not os.path.isdir(hf_dir):
        print(f"ERROR: HuggingFace model directory not found: {hf_dir}", file=sys.stderr)
        print("Run create_model.py first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(ov_dir, exist_ok=True)

    # Task must be explicit for local directories (auto-detection only works for HF Hub names).
    # text-generation-with-past exports a stateful IR where KV-cache is a state variable
    # – required by ContinuousBatchingPipeline.
    #
    # int4 group-wise quantization requires group_size <= channel_size; our hidden dim is
    # tiny (H=16), so we force per-column quantization with --group-size -1.
    extra = "--group-size -1" if weight_format == "int4" else ""
    cmd = (
        f"optimum-cli export openvino"
        f" --model {hf_dir}"
        f" --task text-generation-with-past"
        f" --weight-format {weight_format}"
        f" {extra}"
        f" {ov_dir}"
    )
    print("\n--- Exporting model to OpenVINO IR ---")
    if not run(cmd):
        print("ERROR: optimum-cli export failed.", file=sys.stderr)
        sys.exit(1)


def compile_tokenizer(hf_dir: str, ov_dir: str) -> None:
    """Convert HuggingFace tokenizer to OpenVINO tokenizer + detokenizer."""
    tokenizer_xml = os.path.join(ov_dir, "openvino_tokenizer.xml")
    detokenizer_xml = os.path.join(ov_dir, "openvino_detokenizer.xml")

    if os.path.isfile(tokenizer_xml) and os.path.isfile(detokenizer_xml):
        print("\nOpenVINO tokenizer already present, skipping conversion.")
        return

    print("\n--- Converting tokenizer to OpenVINO format ---")
    cmd = (
        f"convert_tokenizer"
        f" --with-detokenizer"
        f" -o {ov_dir}"
        f" {hf_dir}"
    )
    if not run(cmd):
        # convert_tokenizer is optional; LLMPipeline can also use the HF tokenizer files.
        print("WARNING: tokenizer conversion failed – LLMPipeline will fall back to the HF tokenizer.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dummy LLM to OpenVINO IR")
    parser.add_argument("--hf-model-dir",  default=DEFAULT_HF_DIR,
                        help="Directory containing the HuggingFace model (output of create_model.py)")
    parser.add_argument("--ov-model-dir",  default=DEFAULT_OV_DIR,
                        help="Output directory for the OpenVINO IR model")
    parser.add_argument("--weight-format", default="fp32",
                        choices=["fp32", "fp16", "int8", "int4"],
                        help="Weight precision for the exported IR (default: fp32)")
    args = parser.parse_args()

    print(f"HF model dir   : {args.hf_model_dir}")
    print(f"OV output dir  : {args.ov_model_dir}")
    print(f"Weight format  : {args.weight_format}")

    export_model(args.hf_model_dir, args.ov_model_dir, args.weight_format)
    compile_tokenizer(args.hf_model_dir, args.ov_model_dir)

    print(f"\nOpenVINO model ready at: {args.ov_model_dir}")
    print("Files:")
    for fname in sorted(os.listdir(args.ov_model_dir)):
        size = os.path.getsize(os.path.join(args.ov_model_dir, fname))
        print(f"  {fname:<45s} {size:>10,} bytes")


if __name__ == "__main__":
    main()
