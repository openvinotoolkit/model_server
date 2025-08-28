#!/bin/bash
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
# install dependencies
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/common/export_models/requirements.txt
rm -rf gorilla
git clone https://github.com/ShishirPatil/gorilla
cd gorilla/berkeley-function-call-leaderboard
git checkout cd9429ccf3d4d04156affe883c495b3b047e6b64
curl -s https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/continuous_batching/accuracy/gorilla.patch | git apply -v
pip install -e . 
curl -L -O https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/${BRANCH_NAME}/demos/common/export_models/export_model.py

# Qwen/Qwen3-8B
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-int4 --weight-format int4 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-int8 --weight-format int8 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B-fp16 --weight-format fp16 --model_repository_path models
# Qwen/Qwen3-4B
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-int4 --weight-format int4 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-int8 --weight-format int8 --model_repository_path models
python export_model.py text_generation --source_model Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B-fp16 --weight-format fp16 --model_repository_path models
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
python export_model.py text_generation --source_model NousResearch/Hermes-3-Llama-3.1-8B --model_name NousResearch/Hermes-3-Llama-3.1-8B-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models
curl -L -o models/NousResearch/Hermes-3-Llama-3.1-8B-int8/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_hermes.jinja
python export_model.py text_generation --source_model NousResearch/Hermes-3-Llama-3.1-8B --model_name NousResearch/Hermes-3-Llama-3.1-8B-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models
curl -L -o models/NousResearch/Hermes-3-Llama-3.1-8B-fp16/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_hermes.jinja
# microsoft/Phi-4-mini-instruct
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-int4/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-int8/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models
curl -L -o models/microsoft/Phi-4-mini-instruct-fp16/template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
# mistralai/Mistral-7B-Instruct-v0.3
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --model_name mistralai/Mistral-7B-Instruct-v0.3-int4 --weight-format int4 --config_file_path models/config.json --model_repository_path models --extra_quantization_params "--task text-generation-with-past"
curl -L -o models/mistralai/Mistral-7B-Instruct-v0.3_int4/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_mistral.jinja
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --model_name mistralai/Mistral-7B-Instruct-v0.3-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models --tool_parser mistral --cache_size 2 --extra_quantization_params "--task text-generation-with-past"
curl -L -o models/mistralai/Mistral-7B-Instruct-v0.3_int8/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_mistral.jinja
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --model_name mistralai/Mistral-7B-Instruct-v0.3-fp16 --weight-format fp16 --config_file_path models/config.json --model_repository_path models --tool_parser mistral --cache_size 2 --extra_quantization_params "--task text-generation-with-past"
curl -L -o models/mistralai/Mistral-7B-Instruct-v0.3_fp16/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_mistral.jinja

export OPENAI_BASE_URL=http://localhost:8000/v3
export enable_tool_guided_generation=true
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-8B_int4 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Qwen3-8B_int4
bfcl evaluate --model ovms_model --result-dir Qwen3-8B_int4
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-8B_int8 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Qwen3-8B_int8
bfcl evaluate --model ovms_model --result-dir Qwen3-8B_int8
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-8B_fp16 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Qwen3-8B_fp16
bfcl evaluate --model ovms_model --result-dir Qwen3-8B_fp16
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-4B_int4 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Qwen3-4B_int4
bfcl evaluate --model ovms_model --result-dir Qwen3-4B_int4
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-4B_int8 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Qwen3-4B_int8
bfcl evaluate --model ovms_model --result-dir Qwen3-4B_int8
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-4B_fp16 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Qwen3-4B_fp16
bfcl evaluate --model ovms_model --result-dir Qwen3-4B_fp16
docker stop ovms

docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.1-8B-Instruct_int4 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Llama-3.1-8B-Instruct_int4
bfcl evaluate --model ovms_model --result-dir Llama-3.1-8B-Instruct_int4
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.1-8B-Instruct_int8 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Llama-3.1-8B-Instruct_int8
bfcl evaluate --model ovms_model --result-dir Llama-3.1-8B-Instruct_int8
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.1-8B-Instruct_fp16 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Llama-3.1-8B-Instruct_fp16
bfcl evaluate --model ovms_model --result-dir Llama-3.1-8B-Instruct_fp16
docker stop ovms

docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.2-3B-Instruct_int4 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Llama-3.2-3B-Instruct_int4
bfcl evaluate --model ovms_model --result-dir Llama-3.2-3B-Instruct_int4
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.2-3B-Instruct_int8 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Llama-3.2-3B-Instruct_int8
bfcl evaluate --model ovms_model --result-dir Llama-3.2-3B-Instruct_int8
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.2-3B-Instruct_fp16 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Llama-3.2-3B-Instruct_fp16
bfcl evaluate --model ovms_model --result-dir Llama-3.2-3B-Instruct_fp16
docker stop ovms

docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model NousResearch/Hermes-3-Llama-3.1-8B --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Hermes-3-Llama-3.1-8B_int4
bfcl evaluate --model ovms_model --result-dir Hermes-3-Llama-3.1-8B_int4
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model NousResearch/Hermes-3-Llama-3.1-8B_int8 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Hermes-3-Llama-3.1-8B_int8
bfcl evaluate --model ovms_model --result-dir Hermes-3-Llama-3.1-8B_int8
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model NousResearch/Hermes-3-Llama-3.1-8B_fp16 --tool_parser hermes3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Hermes-3-Llama-3.1-8B_fp16
bfcl evaluate --model ovms_model --result-dir Hermes-3-Llama-3.1-8B_fp16
docker stop ovms

docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model microsoft/Phi-4-mini-instruct_int4 --tool_parser phi4 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Phi-4-mini-instruct_int4
bfcl evaluate --model ovms_model --result-dir Phi-4-mini-instruct_int4
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model microsoft/Phi-4-mini-instruct_int8 --tool_parser phi4 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Phi-4-mini-instruct_int8
bfcl evaluate --model ovms_model --result-dir Phi-4-mini-instruct_int8
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model microsoft/Phi-4-mini-instruct_fp16 --tool_parser phi4 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Phi-4-mini-instruct_fp16
bfcl evaluate --model ovms_model --result-dir Phi-4-mini-instruct_fp16
docker stop ovms

docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model mistralai/Mistral-7B-Instruct-v0.3_int4 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Mistral-7B-Instruct-v0.3_int4
bfcl evaluate --model ovms_model --result-dir Mistral-7B-Instruct-v0.3_int4
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model mistralai/Mistral-7B-Instruct-v0.3_int8 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Mistral-7B-Instruct-v0.3_int8
bfcl evaluate --model ovms_model --result-dir Mistral-7B-Instruct-v0.3_int8
docker stop ovms
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model mistralai/Mistral-7B-Instruct-v0.3_fp16 --tool_parser llama3 --model_name ovms_model --enable_tool_guided_generation $enable_tool_guided_generation --cache_size 20 --task text_generation
bfcl generate --model ovms-model --test-category simple,multiple --num-threads 100 -o --result-dir Mistral-7B-Instruct-v0.3_fp16
bfcl evaluate --model ovms_model --result-dir Mistral-7B-Instruct-v0.3_fp16
docker stop ovms