#!/usr/bin/env bash
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
#
# Build the dummy VLM end-to-end:
#   1. Create/reuse virtual environment
#   2. Create the LLaVA HuggingFace model
#   3. Export to OpenVINO IR (image-text-to-text task)
#   4. Convert tokenizer/image-processor to OV format
#
# Usage:
#   ./build_vlm.sh [OPTIONS]
#
# Options:
#   --skip-venv     Use the current Python environment
#   --skip-export   Only create the HF model; skip OV export
#   --weight-format Weight precision: fp16 (default), fp32, int8; int4 not supported on NPU
#   --sequence      Override the target cyclic phrase (every token must be unique)
#   --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
HF_DIR="$SCRIPT_DIR/vlm_hf_model"
OV_DIR="$SCRIPT_DIR/vlm_ov_model"

SKIP_VENV=0
SKIP_EXPORT=0
WEIGHT_FORMAT="fp16"
SEQUENCE_ARG=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-venv)    SKIP_VENV=1 ;;
        --skip-export)  SKIP_EXPORT=1 ;;
        --weight-format) WEIGHT_FORMAT="$2"; shift ;;
        --weight-format=*) WEIGHT_FORMAT="${1#*=}" ;;
        --sequence) SEQUENCE_ARG=("--sequence" "$2"); shift ;;
        --sequence=*) SEQUENCE_ARG=("--sequence" "${1#*=}") ;;
        --help|-h)
            sed -n '/^# Usage/,/^[^#]/{ /^#/{ s/^# \?//; p } }' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

echo "================================================================"
echo " Dummy Cyclic VLM builder"
echo "   HF model output : $HF_DIR"
echo "   OV model output : $OV_DIR"
echo "   Weight format   : $WEIGHT_FORMAT"
echo "================================================================"

# ── Step 1 : Virtual environment ─────────────────────────────────────────────
if [[ $SKIP_VENV -eq 0 ]]; then
    echo ""
    echo "--- [1/4] Creating virtual environment ---"
    python3 -m venv "$VENV_DIR"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
else
    echo ""
    echo "--- [1/4] Using existing Python environment ---"
    if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
    fi
fi

# ── Step 2 : Create HuggingFace VLM model ────────────────────────────────────
echo ""
echo "--- [2/4] Creating HuggingFace VLM model ---"
python3 "$SCRIPT_DIR/create_vlm_model.py" --output-dir "$HF_DIR" "${SEQUENCE_ARG[@]}"

# ── Step 3 : Export to OpenVINO IR ───────────────────────────────────────────
if [[ $SKIP_EXPORT -eq 0 ]]; then
    echo ""
    echo "--- [3/4] Exporting to OpenVINO IR ---"

    # int4 is not supported by the NPU compiler (UnrollExpandDMA pass fails); use fp16 for NPU.
    # int4 group-size must be <= channel size; our hidden dim is tiny so fall back
    # to per-column quantization.
    EXTRA=""
    [[ "$WEIGHT_FORMAT" == "int4" ]] && EXTRA="--group-size -1"

    optimum-cli export openvino \
        --model "$HF_DIR" \
        --task image-text-to-text \
        --weight-format "$WEIGHT_FORMAT" \
        $EXTRA \
        "$OV_DIR"

    # Convert tokenizer + image processor if not already produced by optimum-cli
    if [[ ! -f "$OV_DIR/openvino_tokenizer.xml" ]]; then
        echo "Converting tokenizer to OpenVINO format..."
        convert_tokenizer --with-detokenizer -o "$OV_DIR" "$HF_DIR" || \
            echo "WARNING: tokenizer conversion failed – VLMPipeline will use the HF tokenizer."
    fi

    echo ""
    echo "OV model files:"
    ls -lh "$OV_DIR"/*.xml 2>/dev/null || true

    echo ""
    echo "--- [4/4] Creating ZIP archive ---"
    ZIP_FILE="${OV_DIR}.zip"
    rm -f "$ZIP_FILE"
    (cd "$(dirname "$OV_DIR")" && zip -r "$(basename "$OV_DIR").zip" "$(basename "$OV_DIR")" -x '*.pyc')
    echo "Archive: $ZIP_FILE"
fi

echo ""
echo "================================================================"
echo " Build complete!"
echo "   HuggingFace VLM model : $HF_DIR"
[[ $SKIP_EXPORT -eq 0 ]] && echo "   OpenVINO IR VLM model : $OV_DIR"
echo ""
echo " Quick test with openvino_genai:"
echo "   python3 -c \""
echo "   import numpy as np, openvino as ov, openvino_genai as g"
echo "   p = g.VLMPipeline('$OV_DIR', 'CPU')"
echo "   img = ov.Tensor(np.zeros((224, 224, 3), dtype=np.uint8))"
echo "   print(p.generate('What do you see?', images=[img], max_new_tokens=30))"
echo "   \""
echo "================================================================"
