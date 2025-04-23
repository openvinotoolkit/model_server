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
export OPENAI_COMPATIBLE_API_URL=http://localhost:8000/v3
export OPENAI_COMPATIBLE_API_KEY="unused"
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
git checkout 4471ad311e620ed6cf3a0419d8ba6f18f8fb1cb3  # https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/625
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


### 5. Results

Results:
```
openai_compatible (model_version=OpenGVLab/InternVL2_5-8B,max_retries=1), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 1
| Tasks  |Version|Filter|n-shot|       Metric       |   |  Value  |   |Stderr|
|--------|-------|------|-----:|--------------------|---|--------:|---|------|
|mme     |Yaml   |none  |     0|mme_cognition_score |↑  | 600.3571|±  |   N/A|
|mme     |Yaml   |none  |     0|mme_perception_score|↑  |1618.2984|±  |   N/A|
|mmmu_val|      0|none  |     0|mmmu_acc            |↑  |   0.5322|±  |   N/A|

```




> **Note:** The same procedure can be used to validate vLLM component. The only needed change would be updating base_url including replacing `/v3/` with `/v1/`.  



