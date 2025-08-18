#!/bin/bash

# install dependencies
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/requirements.txt

git clone https://github.com/ShishirPatil/gorilla
cd gorilla/berkeley-function-call-leaderboard
git checkout cd9429ccf3d4d04156affe883c495b3b047e6b64
git apply -v ../../demos/continuous_batching/accuracy/gorilla.patch
pip install -e . 
cd ../../
cd demos/common/export_model
# Qwen/Qwen3-8B
python ../../demos/common/export_models/export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-int4 --weight-format int4 --model_repository_path models --tool_parser hermes3
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-int8 --weight-format int8 --model_repository_path models --tool_parser hermes3
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-fp16 --weight-format fp16 --model_repository_path models --tool_parser hermes3
# Qwen/Qwen3-4B
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-int4 --weight-format int4 --model_repository_path models --tool_parser hermes3
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-int8 --weight-format int8 --model_repository_path models --tool_parser hermes3
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-fp16 --weight-format fp16 --model_repository_path models --tool_parser hermes3
# meta-llama/Llama-3.1-8B-Instruct
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct-int4 --weight-format int4 --model_repository_path models
curl -L -o models/meta-llama/Llama-3.1-8B-Instruct-int4/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.1_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct-int8 --weight-format int8 --model_repository_path models
curl -L -o models/meta-llama/Llama-3.1-8B-Instruct-int8/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.1_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct-fp16 --weight-format fp16 --model_repository_path models
curl -L -o models/meta-llama/Llama-3.1-8B-Instruct-fp16/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.1_json.jinja
# meta-llama/Llama-3.2-3B-Instruct
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
curl -L -o models/meta-llama/Llama-3.2-3B-Instruct-int4/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.2_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models
    curl -L -o models/meta-llama/Llama-3.2-3B-Instruct-int8/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.2_json.jinja
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models
curl -L -o models/meta-llama/Llama-3.2-3B-Instruct-fp16/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.2_json.jinja
# NousResearch/Hermes-3-Llama-3.1-8B
python export_model.py text_generation --source_model NousResearch/Hermes-3-Llama-3.1-8B --model_name NousResearch/Hermes-3-Llama-3.1-8B-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
curl -L -o models/NousResearch/Hermes-3-Llama-3.1-8B-int4/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_hermes.jinja
# microsoft/Phi-4-mini-instruct
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-int4/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-int8/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-fp16/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja

