# Testing LLM serving accuracy {#ovms_demos_continuous_batching_accuracy}

This guide shows how to access to LLM model over serving endpoint. 

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework provides a convenient method of evaluating the quality of the model exposed over OpenAI API.
It reports end to end quality of served model from the client application point of view. 

## Preparing the lm-evaluation-harness framework 

Set extra index url for CPU-only dependency installation:

::::{tab-set}
:::{tab-item} Bash
:sync: bash
```bash
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
```
:::

:::{tab-item} Windows Command Line
:sync: cmd
```bat
set PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
```
:::

:::{tab-item} Windows PowerShell
:sync: powershell
```powershell
$env:PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
```
:::
::::

Install the framework via pip:
```console
pip3 install lm_eval[api] langdetect immutabledict
```

## Exporting the models
```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
pip3 install -U -r demos/common/export_models/requirements.txt
mkdir models 
python demos/common/export_models/export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config.json --model_repository_path models
python demos/common/export_models/export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config.json --model_repository_path models
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

## Running the tests

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


> **Note** The same procedure can be used to validate vLLM component. The only needed change would be updating base_url including replacing `/v3/` with `/v1/`.  
