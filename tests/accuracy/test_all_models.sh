#!/bin/bash -x
#
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

BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
# install dependencies
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/common/export_models/requirements.txt
rm -rf gorilla
git clone https://github.com/ShishirPatil/gorilla
cd gorilla/berkeley-function-call-leaderboard
git checkout cd9429ccf3d4d04156affe883c495b3b047e6b64
curl -s https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/continuous_batching/accuracy/gorilla.patch | git apply -v
pip install -e . 
cd ../..
curl -L -O https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/common/export_models/export_model.py
mkdir -p models

# Qwen/Qwen3-8B
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-int4 --weight-format int4 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-int8 --weight-format int8 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-fp16 --weight-format fp16 --model_repository_path models
# Qwen/Qwen3-4B
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-int4 --weight-format int4 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-int8 --weight-format int8 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-fp16 --weight-format fp16 --model_repository_path models
# Qwen/Qwen3-1.7B
python export_model.py text_generation --source_model Qwen/Qwen3-1.7B --model_name Qwen/Qwen3-1.7B-int4 --weight-format int4 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-1.7B --model_name Qwen/Qwen3-1.7B-int8 --weight-format int8 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-1.7B --model_name Qwen/Qwen3-1.7B-fp16 --weight-format fp16 --model_repository_path models
# Qwen/Qwen3-0.6B
python export_model.py text_generation --source_model Qwen/Qwen3-0.6B --model_name Qwen/Qwen3-0.6B-int4 --weight-format int4 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-0.6B --model_name Qwen/Qwen3-0.6B-int8 --weight-format int8 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-0.6B --model_name Qwen/Qwen3-0.6B-fp16 --weight-format fp16 --model_repository_path models
# meta-llama/Llama-3.1-8B-Instruct
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct-int4 --weight-format int4 --model_repository_path models
curl -L -o models/meta-llama/Llama-3.1-8B-Instruct-int4/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.1_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct-int8 --weight-format int8 --model_repository_path models
curl -L -o models/meta-llama/Llama-3.1-8B-Instruct-int8/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.1_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct-fp16 --weight-format fp16 --model_repository_path models
curl -L -o models/meta-llama/Llama-3.1-8B-Instruct-fp16/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.1_json.jinja
# meta-llama/Llama-3.2-3B-Instruct
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
curl -L -o models/meta-llama/Llama-3.2-3B-Instruct-int4/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.2_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models
    curl -L -o models/meta-llama/Llama-3.2-3B-Instruct-int8/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.2_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models
curl -L -o models/meta-llama/Llama-3.2-3B-Instruct-fp16/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.2_json.jinja
# NousResearch/Hermes-3-Llama-3.1-8B
python export_model.py text_generation --source_model NousResearch/Hermes-3-Llama-3.1-8B --model_name NousResearch/Hermes-3-Llama-3.1-8B-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
curl -L -o models/NousResearch/Hermes-3-Llama-3.1-8B-int4/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_hermes.jinja
python export_model.py text_generation --source_model NousResearch/Hermes-3-Llama-3.1-8B --model_name NousResearch/Hermes-3-Llama-3.1-8B-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models
curl -L -o models/NousResearch/Hermes-3-Llama-3.1-8B-int8/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_hermes.jinja
python export_model.py text_generation --source_model NousResearch/Hermes-3-Llama-3.1-8B --model_name NousResearch/Hermes-3-Llama-3.1-8B-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models
curl -L -o models/NousResearch/Hermes-3-Llama-3.1-8B-fp16/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_hermes.jinja
# microsoft/Phi-4-mini-instruct
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-int4/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-int8/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-fp16/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
# mistralai/Mistral-7B-Instruct-v0.3
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --model_name mistralai/Mistral-7B-Instruct-v0.3-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models --extra_quantization_params "--task text-generation-with-past"
curl -L -o models/mistralai/Mistral-7B-Instruct-v0.3-int4/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_mistral_parallel.jinja
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --model_name mistralai/Mistral-7B-Instruct-v0.3-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models --tool_parser mistral --cache_size 2 --extra_quantization_params "--task text-generation-with-past"
curl -L -o models/mistralai/Mistral-7B-Instruct-v0.3-int8/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_mistral_parallel.jinja
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --model_name mistralai/Mistral-7B-Instruct-v0.3-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models --tool_parser mistral --cache_size 2 --extra_quantization_params "--task text-generation-with-past"
curl -L -o models/mistralai/Mistral-7B-Instruct-v0.3-fp16/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_mistral_parallel.jinja

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
    --cache_size 20 --task text_generation
    
    sleep 20

    local result_dir="${model_name}-${precision}${enable_tool_guided_generation}"
    # use short model name
    result_dir=$(echo "$result_dir" | awk -F'/' '{print $NF}')
    echo "Result directory: $result_dir"
    sleep 10
    export OPENAI_BASE_URL=http://localhost:8000/v3
    export OPENAI_API_KEY="notused"
    export TOOL_CHOICE=auto
    bfcl generate --model ovms-model --test-category simple,multiple,parallel,irrelevance,multi_turn_base --num-threads 100 --result-dir $result_dir -o
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



