# Testing LLM serving accuracy

This guide shows how to access to LLM model over serving endpoint. 

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework provides a convenient method of evaluating the quality of the model exposed over OpenAI API.
It reports end to end quality of served model from the client application point of view. 

## Preparing the lm-evaluation-harness framework 

Install the framework via:
```bash
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
pip3 install lm_eval[api]
```

## Exporting the model and starting the model server

Follow the procedure to export the model and start the model server from [text generation demo](../README.md)

## Running the tests

```bash
lm-eval --model local-chat-completions --tasks gsm8k --model_args model=meta-llama/Meta-Llama-3-8B-Instruct,base_url=http://localhost:8000/v3/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False --verbosity DEBUG  --log_samples --output_path test/ --seed 1 --apply_chat_template --limit 100

local-chat-completions (model=meta-llama/Meta-Llama-3-8B-Instruct,base_url=http://localhost:8000/v3/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: 100.0, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.62|±  |0.0488|
|     |       |strict-match    |     5|exact_match|↑  | 0.17|±  |0.0378|
```

While testing the non chat model and `completion` endpoint, the command would look like this:

```bash
lm-eval --model local-completions --tasks gsm8k --model_args model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v3/completions,num_concurrent=1,max_retries=3,tokenized_requests=False --verbosity DEBUG  --log_samples --output_path results/ --seed 1 --limit 100

local-completions (model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v3/completions,num_concurrent=1,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: 100.0, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.43|±  |0.0498|
|     |       |strict-match    |     5|exact_match|↑  | 0.43|±  |0.0498|
```

> **Note** The same procedure can be used to validate vLLM component. The only needed change would be updating base_url including replacing `/v3/` with `/v1/`.  

## Limitations
OpenVINO Model Server has partial implementation of `logprobs` parameter so metrics based on `loglikelihood` can't be calculated.
