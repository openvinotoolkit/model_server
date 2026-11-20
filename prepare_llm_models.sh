#!/bin/bash
#
# Copyright 2024 Intel Corporation
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

set -e

prepare_composer_models() (
  repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  onnx_dir=${COMPOSER_ONNX_DIR:-$repo_root/demos/using_onnx_model/python/workspace/resnet50-onnx/1}
  paddle_demo="$repo_root/demos/classification_using_paddlepaddle_model/python"
  paddle_dir=${COMPOSER_PADDLE_DIR:-$paddle_demo/model/1}
  onnx_model="$onnx_dir/resnet50-caffe2-v1-9.onnx"
  stage=$(mktemp -d)
  onnx_download=
  trap 'rm -rf -- "$stage"; if [[ -n $onnx_download ]]; then rm -f -- "$onnx_download"; fi' EXIT

  mkdir -p "$onnx_dir" "$paddle_dir"
  if [[ ! -s $onnx_model ]]; then
    onnx_download=$(mktemp "$onnx_dir/.resnet50.XXXXXX")
    curl --fail --location --connect-timeout 10 --max-time 120 \
      https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx \
      -o "$onnx_download"
    test -s "$onnx_download"
    mv -- "$onnx_download" "$onnx_model"
    onnx_download=
  else
    echo "Reusing ONNX model in $onnx_dir"
  fi
  if [[ ! -s $paddle_dir/inference.pdmodel || ! -s $paddle_dir/inference.pdiparams ]]; then
    (cd "$stage" && timeout 120s python3 "$paddle_demo/download_model.py")
    install -m 644 "$stage/model/1/inference.pdmodel" "$stage/model/1/inference.pdiparams" "$paddle_dir/"
  else
    echo "Reusing PaddlePaddle model in $paddle_dir"
  fi
  chmod a+rx "$onnx_dir" "$(dirname "$onnx_dir")" "$paddle_dir" "$(dirname "$paddle_dir")"
  chmod a+r "$onnx_model" "$paddle_dir/inference.pdmodel" "$paddle_dir/inference.pdiparams"
)

if [[ -z ${1:-} ]]; then
  echo "Usage: $0 <llm-model-directory> [docker]" >&2
  exit 1
fi

if [[ ${2:-} != docker ]]; then
  prepare_composer_models
fi

TOKENIZER_FILE="openvino_tokenizer.bin"
LEGACY_MODEL_FILE="1/model.bin"
EMBEDDING_MODEL="OpenVINO/bge-base-en-v1.5-int8-ov"
RERANK_MODEL="OpenVINO/bge-reranker-base-int8-ov"
TTS_MODEL="OpenVINO/Kokoro-82M-int8-ov"
TEXT_GENERATION_MODEL="OpenVINO/LFM2.5-350M-int8-ov" # TODO: change this model to int4 when it will be public on HF
# Dummy cyclic models used by LLM/VLM tests instead of real downloaded models;
DUMMY_LLM_MODEL="mzeglars/dummy-cyclic-gpt2-ov"
DUMMY_VLM_MODEL="mzeglars/dummy-cyclic-llava-ov"
STT_MODEL="OpenVINO/whisper-tiny-int4-ov"

# Models for tools testing. Only tokenizers are downloaded.
QWEN3_MODEL="OpenVINO/Qwen3-8B-int4-ov"
LLAMA3_MODEL="OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov"
HERMES3_MODEL="NousResearch/Hermes-3-Llama-3.1-8B"
PHI4_MODEL="OpenVINO/Phi-4-reasoning-int4-ov"
MISTRAL_MODEL="OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov"
GPT_OSS_MODEL="OpenVINO/gpt-oss-20b-int4-ov"
DEVSTRAL_MODEL="unsloth/Devstral-Small-2507"
LFM2_MODEL="OpenVINO/LFM2-24B-A2B-int4-ov"
LFM25_MODEL="OpenVINO/LFM2.5-8B-A1B-int4-ov"
GEMMA4_MODEL="OpenVINO/gemma-4-E4B-it-int4-ov"
MINICPM5_MODEL="OpenVINO/MiniCPM5-1B-int4-ov"
ONYX_MODEL="OpenVINO/Muse-Glimmer-30B-int4-ov"

if [ "$(python3 -c 'import sys; print(sys.version_info[1])')" -le "8" ]; then echo "Prepare models with python > 3.8."; exit 1 ; fi

echo "Downloading LLM testing models to directory $1"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"
if [ "${2:-}" = "docker" ]; then
    export PATH=$PATH:/opt/intel/openvino/python/bin
    python3 -m pip install sentence_transformers==5.3.0
else
    python3 -m venv .venv
    . .venv/bin/activate
    pip3 install -U pip
    pip3 install -U sentence_transformers==5.3.0 openvino-tokenizers
fi
mkdir -p $1

# Dummy models are still actively being iterated on - always re-download a fresh copy
# instead of skipping when already present, so local/CI caches can't go stale.
# if [ -f "$1/$DUMMY_LLM_MODEL/$TOKENIZER_FILE" ]; then
#   echo "Model file $1/$DUMMY_LLM_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
# else
  pip3 install --upgrade typer==0.25.1
  hf download "$DUMMY_LLM_MODEL" --local-dir "$1/$DUMMY_LLM_MODEL"
# fi
if [ ! -f "$1/$DUMMY_LLM_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Model file $1/$DUMMY_LLM_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

# Dummy models are still actively being iterated on - always re-download a fresh copy
# instead of skipping when already present, so local/CI caches can't go stale.
# if [ -f "$1/$DUMMY_VLM_MODEL/$TOKENIZER_FILE" ]; then
#   echo "Model file $1/$DUMMY_VLM_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
# else
  hf download "$DUMMY_VLM_MODEL" --local-dir "$1/$DUMMY_VLM_MODEL"  # typer already pinned above
# fi
if [ ! -f "$1/$DUMMY_VLM_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Model file $1/$DUMMY_VLM_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$TEXT_GENERATION_MODEL/openvino_model.xml" ]; then
  echo "Model file $1/$TEXT_GENERATION_MODEL/openvino_model.xml exists. Skipping downloading models."
else
  hf download "$TEXT_GENERATION_MODEL" --local-dir "$1/$TEXT_GENERATION_MODEL"
fi
if [ ! -f "$1/$TEXT_GENERATION_MODEL/openvino_model.xml" ]; then
  echo "[ERROR] Model file $1/$TEXT_GENERATION_MODEL/openvino_model.xml does not exist."
  exit 1
fi

if [ -f "$1/$TTS_MODEL/openvino_model.xml" ]; then
  echo "Model file $1/$TTS_MODEL/openvino_model.xml exists. Skipping downloading models."
else
  hf download "$TTS_MODEL" --local-dir "$1/$TTS_MODEL"
fi
if [ ! -f "$1/$TTS_MODEL/openvino_model.xml" ]; then
  echo "[ERROR] Model file $1/$TTS_MODEL/openvino_model.xml does not exist."
  exit 1
fi

if [ -f "$1/$STT_MODEL/$TOKENIZER_FILE" ]; then
  echo "Model file $1/$STT_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  hf download "$STT_MODEL" --local-dir "$1/$STT_MODEL"
fi
if [ ! -f "$1/$STT_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Model file $1/$STT_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi
if [ -f "$1/$EMBEDDING_MODEL/$TOKENIZER_FILE" ]; then
  echo "Model file $1/$EMBEDDING_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  hf download "$EMBEDDING_MODEL" --local-dir "$1/$EMBEDDING_MODEL"
fi
if [ ! -f "$1/$EMBEDDING_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Model file $1/$EMBEDDING_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$RERANK_MODEL/$TOKENIZER_FILE" ]; then
  echo "Model file $1/$RERANK_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  hf download "$RERANK_MODEL" --local-dir "$1/$RERANK_MODEL"
fi
if [ ! -f "$1/$RERANK_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Model file $1/$RERANK_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$QWEN3_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$QWEN3_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$QWEN3_MODEL
  convert_tokenizer $QWEN3_MODEL --with_detokenizer -o $1/$QWEN3_MODEL
fi
if [ ! -f "$1/$QWEN3_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$QWEN3_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$LLAMA3_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$LLAMA3_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$LLAMA3_MODEL
  convert_tokenizer $LLAMA3_MODEL --with_detokenizer -o $1/$LLAMA3_MODEL
fi
if [ ! -f "$1/$LLAMA3_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$LLAMA3_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$HERMES3_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$HERMES3_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$HERMES3_MODEL
  convert_tokenizer $HERMES3_MODEL --with_detokenizer -o $1/$HERMES3_MODEL
fi
if [ ! -f "$1/$HERMES3_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$HERMES3_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$PHI4_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$PHI4_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$PHI4_MODEL
  convert_tokenizer $PHI4_MODEL --with_detokenizer -o $1/$PHI4_MODEL
fi
if [ ! -f "$1/$PHI4_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$PHI4_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$MISTRAL_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$MISTRAL_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$MISTRAL_MODEL
  convert_tokenizer $MISTRAL_MODEL --with_detokenizer -o $1/$MISTRAL_MODEL
fi
if [ ! -f "$1/$MISTRAL_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$MISTRAL_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$GPT_OSS_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$GPT_OSS_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$GPT_OSS_MODEL
  convert_tokenizer $GPT_OSS_MODEL --with_detokenizer -o $1/$GPT_OSS_MODEL
fi
if [ ! -f "$1/$GPT_OSS_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$GPT_OSS_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$DEVSTRAL_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$DEVSTRAL_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$DEVSTRAL_MODEL
  convert_tokenizer $DEVSTRAL_MODEL --with_detokenizer -o $1/$DEVSTRAL_MODEL
fi
if [ ! -f "$1/$DEVSTRAL_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$DEVSTRAL_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$LFM2_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$LFM2_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$LFM2_MODEL
  convert_tokenizer $LFM2_MODEL --with_detokenizer -o $1/$LFM2_MODEL
fi
if [ ! -f "$1/$LFM2_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$LFM2_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi
if [ -f "$1/$LFM25_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$LFM25_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$LFM25_MODEL
  convert_tokenizer $LFM25_MODEL --with_detokenizer -o $1/$LFM25_MODEL
fi
if [ ! -f "$1/$LFM25_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$LFM25_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi
if [ -f "$1/$GEMMA4_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$GEMMA4_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$GEMMA4_MODEL
  convert_tokenizer $GEMMA4_MODEL --with_detokenizer -o $1/$GEMMA4_MODEL
fi
if [ ! -f "$1/$GEMMA4_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$GEMMA4_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$MINICPM5_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$MINICPM5_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$MINICPM5_MODEL
  convert_tokenizer $MINICPM5_MODEL --with_detokenizer -o $1/$MINICPM5_MODEL
fi
if [ ! -f "$1/$MINICPM5_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$MINICPM5_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi

if [ -f "$1/$ONYX_MODEL/$TOKENIZER_FILE" ]; then
  echo "Models file $1/$ONYX_MODEL/$TOKENIZER_FILE exists. Skipping downloading models."
else
  mkdir -p $1/$ONYX_MODEL
  convert_tokenizer $ONYX_MODEL --with_detokenizer -o $1/$ONYX_MODEL
fi
if [ ! -f "$1/$ONYX_MODEL/$TOKENIZER_FILE" ]; then
  echo "[ERROR] Models file $1/$ONYX_MODEL/$TOKENIZER_FILE does not exist."
  exit 1
fi
