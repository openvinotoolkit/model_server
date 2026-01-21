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

BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
# install dependencies
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/common/export_models/requirements.txt
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
# openai/gpt-oss-20b
python export_model.py text_generation --source_model openai/gpt-oss-20b --model_name openai/gpt-oss-20b-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
cp ../extras/chat_template_examples/chat_template_gpt_oss.jinja models/openai/gpt-oss-20b-int4/chat_template.jinja

# Qwen/Qwen3-Coder-30B-Instruct
python export_model.py text_generation --source_model Qwen/Qwen3-Coder-30B-Instruct --model_name Qwen/Qwen3-Coder-30B-Instruct-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
cp ../extras/chat_template_examples/chat_template_qwen3coder_instruct.jinja models/Qwen/Qwen3-Coder-30B-Instruct-int4/chat_template.jinja

# devstral
python export_model.py text_generation --source_model unsloth/Devstral-Small-2507 --model_name unsloth/Devstral-Small-2507-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
cp ../extras/chat_template_examples/chat_template_devstral.jinja models/unsloth/Devstral-Small-2507-int4/chat_template.jinja