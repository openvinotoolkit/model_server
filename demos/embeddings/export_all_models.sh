#/bin/bash
# Copyright (c) 2024 Intel Corporation
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
# Execute it in the context of git repository in the demos/embeddings folder
# Add or remove models from the tested_models array to export them

declare -A model_params
model_params["nomic-ai/nomic-embed-text-v1.5"]="--pooling MEAN --extra_quantization_params \"--library sentence_transformers\""
model_params["Alibaba-NLP/gte-large-en-v1.5"]="--pooling CLS --extra_quantization_params \"--library sentence_transformers\""
model_params["BAAI/bge-large-en-v1.5"]="--pooling CLS"
model_params["BAAI/bge-large-zh-v1.5"]="--pooling CLS"
model_params["thenlper/gte-small"]="--pooling CLS"
model_params["Qwen/Qwen3-Embedding-0.6B"]="--pooling LAST"
model_params["sentence-transformers/all-MiniLM-L12-v2"]="--pooling MEAN"
model_params["sentence-transformers/all-distilroberta-v1"]="--pooling MEAN"
model_params["mixedbread-ai/deepset-mxbai-embed-de-large-v1"]="--pooling MEAN"
model_params["intfloat/multilingual-e5-large-instruct"]="--pooling MEAN"
model_params["intfloat/multilingual-e5-large"]="--pooling MEAN"
model_params["sentence-transformers/all-mpnet-base-v2"]="--pooling MEAN"

tested_models=(
    nomic-ai/nomic-embed-text-v1.5
    Alibaba-NLP/gte-large-en-v1.5
    BAAI/bge-large-en-v1.5 
    BAAI/bge-large-zh-v1.5
    thenlper/gte-small
    Qwen/Qwen3-Embedding-0.6B
    sentence-transformers/all-MiniLM-L12-v2
    sentence-transformers/all-distilroberta-v1
    mixedbread-ai/deepset-mxbai-embed-de-large-v1
    intfloat/multilingual-e5-large-instruct
    intfloat/multilingual-e5-large
    sentence-transformers/all-mpnet-base-v2
)

mkdir models

for i in "${tested_models[@]}"; do
    echo "$i"
    eval "python export_model.py embeddings_ov --source_model \"$i\" --weight-format int8 ${model_params[$i]} --config_file_path models/config_all.json"
done
