export MODEL=$1
export PRECISION=$2


docker stop ovms 2>/dev/null
docker run -d --name ovms --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path /models --source_model ${model_name}-${precision} \
--tool_parser ${tool_parser} --model_name ovms-model \
--cache_size 0 --task text_generation
    
echo wait for model server to be ready
while [ "$(curl -s http://localhost:8000/v3/models | jq -r '.data[0].id')" != "${model_name}-${precision}" ] ; do echo waiting for LLM model; sleep 1; done
echo Server is ready

export result_dir="${MODEL}-${PRECISION}"
# use short model name
result_dir=$(echo "$result_dir" | awk -F'/' '{print $NF}')
echo "Result directory: $result_dir"
export OPENAI_BASE_URL=http://localhost:8000/v3

bfcl generate --model ovms-model --run-ids --result-dir $result_dir -o
bfcl evaluate --model ovms-model --result-dir $result_dir --score-dir ${result_dir}_score --partial-eval
