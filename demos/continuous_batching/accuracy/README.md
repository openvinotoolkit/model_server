# Testing LLM and VLM serving accuracy {#ovms_demos_continuous_batching_accuracy}

This guide shows how to access to LLM and VLM model over serving endpoint. 

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework provides a convenient method of evaluating the quality of the model exposed over OpenAI API.
It reports end to end quality of served model from the client application point of view. 

## Preparing the lm-evaluation-harness framework 

Install the framework via pip:
```console
pip3 install --extra-index-url "https://download.pytorch.org/whl/cpu" lm_eval[api] langdetect immutabledict dotenv openai
```

## Exporting the models
```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
pip3 install -U -r demos/common/export_models/requirements.txt
mkdir models 
python demos/common/export_models/export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config.json --model_repository_path models
python demos/common/export_models/export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config.json --model_repository_path models
python demos/common/export_models/export_model.py text_generation --source_model OpenGVLab/InternVL2_5-8B --weight-format fp16 --config_file_path models/config.json --model_repository_path models
python demos/common/export_models/export_model.py text_generation --source_model Qwen/Qwen3-8B --model_name openvino-qwen3-8b-int8 --weight-format int8 --config_file_path models/config.json --model_repository_path models --tools_model_type qwen3 --overwrite_models --enable_prefix_caching
```

## Starting the model server

### With Docker
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config.json
```

### On Baremetal
```bat
ovms --rest_port 8000 --config_path ./models/config.json
```

## Running the tests for LLM models

```console
lm-eval --model local-chat-completions --tasks gsm8k --model_args model=meta-llama/Meta-Llama-3-8B-Instruct,base_url=http://localhost:8000/v3/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False --verbosity DEBUG  --log_samples --output_path test/ --seed 1 --apply_chat_template --limit 100

local-chat-completions (model=meta-llama/Meta-Llama-3-8B-Instruct,base_url=http://localhost:8000/v3/chat/completions,num_concurrent=10,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: 100.0, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.62|±  |0.0488|
|     |       |strict-match    |     5|exact_match|↑  | 0.17|±  |0.0378|
```

While testing the non chat model and `completion` endpoint, the command would look like this:

```console
lm-eval --model local-completions --tasks gsm8k --model_args model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v3/completions,num_concurrent=1,max_retries=3,tokenized_requests=False --verbosity DEBUG  --log_samples --output_path results/ --seed 1 --limit 100

local-completions (model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v3/completions,num_concurrent=10,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: 100.0, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.43|±  |0.0498|
|     |       |strict-match    |     5|exact_match|↑  | 0.43|±  |0.0498|
```

Other examples are below:

```console
lm-eval --model local-chat-completions --tasks leaderboard_ifeval --model_args model=meta-llama/Meta-Llama-3-8B-Instruct,base_url=http://localhost:8000/v3/chat/completions,num_concurrent=10,max_retries=3,tokenized_requests=False --verbosity DEBUG --log_samples --output_path test/ --seed 1 --limit 100 --apply_chat_template  
```

```console
lm-eval --model local-completions --tasks wikitext --model_args model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v3/completions,num_concurrent=10,max_retries=3,tokenized_requests=False --verbosity DEBUG --log_samples --output_path test/ --seed 1 --limit 100
```

## Running the tests for VLM models

Use [lmms-eval project](https://github.com/EvolvingLMMs-Lab/lmms-eval) - mme and mmmu_val tasks. 

```bash
export OPENAI_BASE_URL=http://localhost:8000/v3
export OPENAI_API_KEY="unused"
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
git checkout f64dfa5fd063e989a0a665d2fd0615df23888c83
pip install -e . --extra-index-url "https://download.pytorch.org/whl/cpu"
python -m lmms_eval \
    --model openai_compatible \
    --model_args model_version=OpenGVLab/InternVL2_5-8B,max_retries=1 \
    --tasks mme,mmmu_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix openai_compatible \
    --output_path ./logs
```

**Results example:**

```text
openai_compatible (model_version=OpenGVLab/InternVL2_5-8B,max_retries=1), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
| Tasks  |Version|Filter|n-shot|       Metric       |   |  Value  |   |Stderr|
|--------|-------|------|-----:|--------------------|---|--------:|---|------|
|mme     |Yaml   |none  |     0|mme_cognition_score |↑  | 600.3571|±  |   N/A|
|mme     |Yaml   |none  |     0|mme_perception_score|↑  |1618.2984|±  |   N/A|
|mmmu_val|      0|none  |     0|mmmu_acc            |↑  |   0.5322|±  |   N/A|

```


## Running the tests for agentic models with function calls

Use [Berkeley function call leaderboard ](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)


```bash
git clone https://github.com/ShishirPatil/gorilla
cd gorilla/berkeley-function-call-leaderboard
git checkout cd9429ccf3d4d04156affe883c495b3b047e6b64
curl -s https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/continuous_batching/accuracy/gorilla.patch | git apply -v
pip install -e . 
```
The commands below assumes the models is deployed with the name `ovms-model`. It must match the name set in the `bfcl_eval/constants/model_config.py`.
```bash
export OPENAI_BASE_URL=http://localhost:8000/v3
bfcl generate --model ovms-model --test-category simple,multiple --temperature 0.0 --num-threads 100 -o --result-dir model_name_dir
bfcl evaluate --model ovms-model --result-dir model_name_dir 
```

Alternatively, use the model name `ovms-model-stream` to run the tests with stream requests. The results should be the same.
```bash
export OPENAI_BASE_URL=http://localhost:8000/v3
bfcl generate --model ovms-model-stream --test-category simple,multiple --temperature 0.0 --num-threads 100 -o --result-dir model_name_dir
bfcl evaluate --model ovms-model-stream --result-dir model_name_dir 
```

**Analyzing results**
The output artifacts will be stored in `result` and `scores`. For example:

```text
cat score/openvino-qwen3-8b-int4-FC/BFCL_v3_simple_score.json | head -1
{"accuracy": 0.95, "correct_count": 380, "total_count": 400}
```
Those results can be compared with the reference from the [berkeley leaderbaord](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard).

---

> **Note:** The same procedure can be used to validate vLLM component. The only needed change would be updating base_url including replacing `/v3/` with `/v1/`.  



