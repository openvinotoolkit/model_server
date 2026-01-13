#!/bin/bash -x
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

run_model_test() {
    local model_name=$1
    local precision=$2
    local tool_parser=$3
    local enable_tool_guided_generation=${4:-false}
    set -x
    docker stop ovms 2>/dev/null
    docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
    --rest_port 8000 --model_repository_path /models --source_model ${model_name}-${precision} \
    --tool_parser ${tool_parser} --model_name ovms-model --enable_tool_guided_generation $enable_tool_guided_generation \
    --cache_size 0 --task text_generation
    
    echo wait for model server to be ready
    while [ "$(curl -s http://localhost:8000/v3/models | jq -r '.data[0].id')" != "${model_name}-${precision}" ] ; do echo waiting for LLM model; sleep 1; done
    echo Server is ready

    local result_dir="${model_name}-${precision}${enable_tool_guided_generation}"
    # use short model name
    result_dir=$(echo "$result_dir" | awk -F'/' '{print $NF}')
    echo "Result directory: $result_dir"
    sleep 10
    export OPENAI_BASE_URL=http://localhost:8000/v3
    export OPENAI_API_KEY="notused"
    export TOOL_CHOICE=auto
    bfcl generate --model ovms-model --test-category simple_python,multiple,parallel,irrelevance,multi_turn_base --num-threads 100 --result-dir $result_dir -o
    bfcl generate --model ovms-model --test-category multi_turn_base --num-threads 10 --result-dir $result_dir -o
    bfcl evaluate --model ovms-model --result-dir $result_dir --score-dir ${result_dir}_score
}

# Model configurations
declare -A models=(
    ["Qwen/Qwen3-8B"]="hermes3"
    ["Qwen/Qwen3-4B"]="hermes3"
    ["Qwen/Qwen3-1.7B"]="hermes3"
    ["Qwen/Qwen3-0.6B"]="hermes3"
    ["meta-llama/Llama-3.1-8B-Instruct"]="llama3"
    ["meta-llama/Llama-3.2-3B-Instruct"]="llama3"
    ["NousResearch/Hermes-3-Llama-3.1-8B"]="hermes3"
    ["microsoft/Phi-4-mini-instruct"]="phi4"
    ["mistralai/Mistral-7B-Instruct-v0.3"]="mistral"
)

precisions=("int4" "int8" "fp16")

# Run tests for each model and precision
# enable tool guided generation
for model in "${!models[@]}"; do
    tool_parser="${models[$model]}"
    for precision in "${precisions[@]}"; do
        run_model_test "$model" "$precision" "$tool_parser" "true"
    done
done


# disable tool guided generation
for model in "${!models[@]}"; do
    tool_parser="${models[$model]}"
    for precision in "${precisions[@]}"; do
        run_model_test "$model" "$precision" "$tool_parser" "false"
    done
done

docker stop ovms 2>/dev/null


python sumarize_results.py



