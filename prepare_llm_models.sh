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

if [ -z "$1" ]; then
  echo "Error: No directory specified."
  exit 1
fi

CB_MODEL="facebook/opt-125m"
EMBEDDING_MODEL="thenlper/gte-small"
RERANK_MODEL="BAAI/bge-reranker-base"
VLM_MODEL="OpenGVLab/InternVL2-1B"

# Models for tools testing. Only tokenizers are downloaded.
QWEN3_MODEL="Qwen/Qwen3-8B"
LLAMA3_MODEL="meta-llama/Llama-3.1-8B-Instruct"
HERMES3_MODEL="NousResearch/Hermes-3-Llama-3.1-8B"
PHI4_MODEL="microsoft/Phi-4-mini-instruct"

MODELS=("$CB_MODEL" "$EMBEDDING_MODEL" "$RERANK_MODEL" "$VLM_MODEL" "$QWEN3_MODEL" "$LLAMA3_MODEL" "$HERMES3_MODEL" "$PHI4_MODEL" "$EMBEDDING_MODEL/ov" "$RERANK_MODEL/ov")

all_exist=true
for model in "${MODELS[@]}"; do
  if [ ! -d "$1/$model" ]; then
    all_exist=false
    break
  fi
done

if $all_exist; then
  echo "All model directories exist in $1. Skipping downloading models."
  exit 0
fi

if [ "$(python3 -c 'import sys; print(sys.version_info[1])')" -le "8" ]; then echo "Prepare models with python > 3.8."; exit 1 ; fi

echo "Downloading LLM testing models to directory $1"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"
if [ "$2" = "docker" ]; then
    sed -i '/openvino~=/d' /openvino_tokenizers/pyproject.toml
    python3 -m pip wheel -v --no-deps --wheel-dir wheel /openvino_tokenizers
    python3 -m pip install $(find wheel -name 'openvino_tokenizers*.whl')
    python3 -m pip install "optimum-intel"@git+https://github.com/huggingface/optimum-intel.git nncf sentence_transformers==3.1.1
else
    python3.10 -m venv .venv
    . .venv/bin/activate
    pip3 install -U pip
    pip3 install -U -r demos/common/export_models/requirements.txt
fi
mkdir -p $1

if [ -d "$1/$CB_MODEL" ]; then
  echo "Models directory $1/$CB_MODEL exists. Skipping downloading models."
else
  python3 demos/common/export_models/export_model.py text_generation --source_model "$CB_MODEL" --weight-format int8 --model_repository_path $1
fi

if [ -d "$1/$VLM_MODEL" ]; then
  echo "Models directory $1/$VLM_MODEL exists. Skipping downloading models."
else
  python3 demos/common/export_models/export_model.py text_generation --source_model "$VLM_MODEL" --weight-format int4 --kv_cache_precision u8 --model_repository_path $1
fi

if [ -d "$1/$EMBEDDING_MODEL" ]; then
  echo "Models directory $1/$EMBEDDING_MODEL exists. Skipping downloading models."
else
  python3 demos/common/export_models/export_model.py embeddings --source_model "$EMBEDDING_MODEL" --weight-format int8 --model_repository_path $1
fi

if [ -d "$1/$EMBEDDING_MODEL/ov" ]; then
  echo "Models directory "$1/$EMBEDDING_MODEL/ov" exists. Skipping downloading models."
else
  python3 demos/common/export_models/export_model.py embeddings_ov --source_model "$EMBEDDING_MODEL" --weight-format int8 --model_repository_path $1 --model_name $EMBEDDING_MODEL/ov
fi

if [ -d "$1/$RERANK_MODEL" ]; then
  echo "Models directory $1/$RERANK_MODEL exists. Skipping downloading models."
else
  python3 demos/common/export_models/export_model.py rerank --source_model "$RERANK_MODEL" --weight-format int8 --model_repository_path $1
fi

if [ -d "$1/$RERANK_MODEL/ov" ]; then
  echo "Models directory $1/$RERANK_MODEL/ov exists. Skipping downloading models."
else
  python3 demos/common/export_models/export_model.py rerank_ov --source_model "$RERANK_MODEL" --weight-format int8 --model_repository_path $1 --model_name $RERANK_MODEL/ov
fi

if [ -d "$1/$QWEN3_MODEL" ]; then
  echo "Models directory $1/$QWEN3_MODEL exists. Skipping downloading models."
else
  mkdir -p $1/$QWEN3_MODEL
  convert_tokenizer $QWEN3_MODEL --with_detokenizer -o $1/$QWEN3_MODEL
fi

if [ -d "$1/$LLAMA3_MODEL" ]; then
  echo "Models directory $1/$LLAMA3_MODEL exists. Skipping downloading models."
else
  mkdir -p $1/$LLAMA3_MODEL
  convert_tokenizer $LLAMA3_MODEL --with_detokenizer -o $1/$LLAMA3_MODEL
fi

if [ -d "$1/$HERMES3_MODEL" ]; then
  echo "Models directory $1/$HERMES3_MODEL exists. Skipping downloading models."
else
  mkdir -p $1/$HERMES3_MODEL
  convert_tokenizer $HERMES3_MODEL --with_detokenizer -o $1/$HERMES3_MODEL
fi

if [ -d "$1/$PHI4_MODEL" ]; then
  echo "Models directory $1/$PHI4_MODEL exists. Skipping downloading models."
else
  mkdir -p $1/$PHI4_MODEL
  convert_tokenizer $PHI4_MODEL --with_detokenizer -o $1/$PHI4_MODEL
fi
