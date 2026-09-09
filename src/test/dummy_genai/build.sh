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
# Build the dummy cyclic LLM end-to-end:
#   1. Create Python virtual environment and install dependencies
#   2. Create the HuggingFace model (GPT-2 with crafted weights)
#   3. Export to OpenVINO IR via optimum-cli + convert_tokenizer
#   4. Verify both models generate the expected cyclic sequence
#
# Usage:
#   ./build.sh [OPTIONS]
#
# Options:
#   --skip-venv     Use the current Python environment instead of creating a venv
#   --skip-export   Only create the HF model; skip OpenVINO IR export
#   --skip-verify   Skip the verification step
#   --weight-format Weight precision for the OV export: fp16 (default), fp32, int8; int4 not supported on NPU
#   --sequence      Override the target cyclic sequence (must not have repeated tokens)
#   --help          Show this message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
HF_DIR="$SCRIPT_DIR/hf_model"
OV_DIR="$SCRIPT_DIR/ov_model"

SKIP_VENV=0
SKIP_EXPORT=0
SKIP_VERIFY=0
WEIGHT_FORMAT="fp16"
SEQUENCE_ARG=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-venv)    SKIP_VENV=1 ;;
        --skip-export)  SKIP_EXPORT=1 ;;
        --skip-verify)  SKIP_VERIFY=1 ;;
        --weight-format)
            WEIGHT_FORMAT="$2"; shift ;;
        --weight-format=*)
            WEIGHT_FORMAT="${1#*=}" ;;
        --sequence)
            SEQUENCE_ARG=("--sequence" "$2"); shift ;;
        --sequence=*)
            SEQUENCE_ARG=("--sequence" "${1#*=}") ;;
        --help|-h)
            sed -n '/^# Usage/,/^[^#]/{ /^#/{ s/^# \?//; p } }' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo " Dummy Cyclic LLM builder"
echo "   HF model output : $HF_DIR"
echo "   OV model output : $OV_DIR"
echo "   Weight format   : $WEIGHT_FORMAT"
echo "================================================================"

# ── Step 1 : Virtual environment ─────────────────────────────────────────────
if [[ $SKIP_VENV -eq 0 ]]; then
    echo ""
    echo "--- [1/5] Creating virtual environment ---"
    python3 -m venv "$VENV_DIR"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
    echo "  venv ready: $VENV_DIR"
else
    echo ""
    echo "--- [1/5] Using existing Python environment (--skip-venv) ---"
    # Activate the venv if it exists and we're not already inside it
    if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
    fi
fi

# ── Step 2 : Create HuggingFace model ────────────────────────────────────────
echo ""
echo "--- [2/5] Creating HuggingFace model ---"
python3 "$SCRIPT_DIR/create_model.py" --output-dir "$HF_DIR" "${SEQUENCE_ARG[@]}"

# ── Step 3 : Export to OpenVINO IR ───────────────────────────────────────────
if [[ $SKIP_EXPORT -eq 0 ]]; then
    echo ""
    echo "--- [3/5] Exporting to OpenVINO IR ---"
    # int4 is not supported by the NPU compiler (UnrollExpandDMA pass fails); use fp16 for NPU.
    python3 "$SCRIPT_DIR/export_to_ov.py" \
        --hf-model-dir  "$HF_DIR" \
        --ov-model-dir  "$OV_DIR" \
        --weight-format "$WEIGHT_FORMAT"

    # Ship the model card as the final catalog's README (this is what gets published to HF)
    cp "$SCRIPT_DIR/model_card_llm.md" "$OV_DIR/README.md"
else
    echo ""
    echo "--- [3/5] Skipping OpenVINO export (--skip-export) ---"
fi

# ── Step 4 : Verify ──────────────────────────────────────────────────────────
if [[ $SKIP_VERIFY -eq 0 ]]; then
    echo ""
    echo "--- [4/5] Verifying models ---"
    OV_FLAG=""
    [[ $SKIP_EXPORT -eq 1 ]] && OV_FLAG="--skip-ov"
    python3 "$SCRIPT_DIR/verify_model.py" \
        --hf-model-dir "$HF_DIR" \
        --ov-model-dir "$OV_DIR" \
        $OV_FLAG
else
    echo ""
    echo "--- [4/5] Skipping verification (--skip-verify) ---"
fi

# ── Step 5 : ZIP archive ─────────────────────────────────────────────────────
if [[ $SKIP_EXPORT -eq 0 ]]; then
    echo ""
    echo "--- [5/5] Creating ZIP archive ---"
    ZIP_FILE="${OV_DIR}.zip"
    rm -f "$ZIP_FILE"
    (cd "$(dirname "$OV_DIR")" && zip -r "$(basename "$OV_DIR").zip" "$(basename "$OV_DIR")" -x '*.pyc')
    echo "Archive: $ZIP_FILE"
else
    echo ""
    echo "--- [5/5] Skipping ZIP (no OV model) ---"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Build complete!"
echo "   HuggingFace model : $HF_DIR"
if [[ $SKIP_EXPORT -eq 0 ]]; then
    echo "   OpenVINO IR model : $OV_DIR"
fi
echo ""
echo " Quick test with openvino_genai:"
echo "   python3 -c \""
echo "   import openvino_genai as g"
echo "   p = g.LLMPipeline('$OV_DIR', 'CPU')"
echo "   print(p.generate('Hello', max_new_tokens=40, do_sample=False))"
echo "   \""
echo "================================================================"
