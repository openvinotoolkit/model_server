# How to serve LLM Models in Speculative Decoding Pipeline{#ovms_demos_continuous_batching_speculative_decoding}

Following [OpenVINO GenAI docs](https://docs.openvino.ai/2025/learn-openvino/llm_inference_guide/genai-guide.html#efficient-text-generation-via-speculative-decoding):
> Speculative decoding (or assisted-generation) enables faster token generation when an additional smaller draft model is used alongside the main model. This reduces the number of infer requests to the main model, increasing performance.
> 
> The draft model predicts the next K tokens one by one in an autoregressive manner. The main model validates these predictions and corrects them if necessary - in case of a discrepancy, the main model prediction is used. Then, the draft model acquires this token and runs prediction of the next K tokens, thus repeating the cycle.

The goal of this sampling method is to reduce latency while keeping the main model accuracy. It gives the biggest gain in low concurrency scenario.

This demo shows how to use speculative decoding in the model serving scenario, by deploying main and draft models in a speculative decoding pipeline in a manner similar to regular deployments with continuous batching.

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

## Model considerations

From the functional perspective both main and draft models must use the same tokenizer, so the tokens from the draft model are correctly matched in the the main model.

From the performance perspective, benefits from speculative decoding are strictly tied to the pair of models used.
For some models, the performance boost is significant, while for others it's rather negligible. Models sizes and precisions also come into play, so optimal setup shall be found empirically.

In this demo we will use:
  - [meta-llama/CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf) as a main model
  - [AMD-Llama-135m](https://huggingface.co/amd/AMD-Llama-135m) as a draft model

both in FP16 precision.

## Model preparation
Here, the original Pytorch LLM models and the tokenizers will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Download export script, install its dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
mkdir models 
```

Run `export_model.py` script to download and quantize the model:

> **Note:** Before downloading the CodeLlama model, access must be requested. Follow the instructions on the [meta-llama/CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf) to request access. When access is granted, create an authentication token in the HuggingFace account -> Settings -> Access Tokens page. Issue the following command and enter the authentication token. Authenticate via `huggingface-cli login`.

```console
python export_model.py text_generation --source_model meta-llama/CodeLlama-7b-hf --draft_source_model amd/AMD-Llama-135m --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config.json --model_repository_path models
```

Draft model inherits all scheduler properties from the main model.

You should have a model folder like below:
```
models
├── config.json
└── meta-llama
    └── CodeLlama-7b-hf
        ├── amd-AMD-Llama-135m
        │   ├── config.json
        │   ├── generation_config.json
        │   ├── openvino_detokenizer.bin
        │   ├── openvino_detokenizer.xml
        │   ├── openvino_model.bin
        │   ├── openvino_model.xml
        │   ├── openvino_tokenizer.bin
        │   ├── openvino_tokenizer.xml
        │   ├── special_tokens_map.json
        │   ├── tokenizer_config.json
        │   ├── tokenizer.json
        │   └── tokenizer.model
        ├── config.json
        ├── generation_config.json
        ├── graph.pbtxt
        ├── openvino_detokenizer.bin
        ├── openvino_detokenizer.xml
        ├── openvino_model.bin
        ├── openvino_model.xml
        ├── openvino_tokenizer.bin
        ├── openvino_tokenizer.xml
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── tokenizer.model

```

## Server Deployment

:::{dropdown} **Deploying with Docker**
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config.json
```

Running above command starts the container with no accelerators support. 
To deploy on devices other than CPU, change `target_device` parameter in `export_model.py` call and follow [AI accelerators guide](../../../docs/accelerators.md) for additionally required docker parameters.
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --config_path ./models/config.json
```
:::

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
    "meta-llama/CodeLlama-7b-hf": {
        "model_version_status": [
            {
                "version": "1",
                "state": "AVAILABLE",
                "status": {
                    "error_code": "OK",
                    "error_message": "OK"
                }
            }
        ]
    }
}
```

## Request Generation

Models used in this demo - `meta-llama/CodeLlama-7b-hf` and `AMD-Llama-135m` are not chat models, so we will use `completions` endpoint to interact with the pipeline.

Below you can see an exemplary unary request (you can switch `stream` parameter to enable streamed response). Compared to calls to regular continuous batching model, this request has additional parameter `num_assistant_tokens` which specifies how many tokens should a draft model generate before main model validates them. 


```console
curl http://localhost:8000/v3/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/CodeLlama-7b-hf",
    "temperature": 0,
    "max_tokens":100,
    "stream":false,
    "prompt": "<s>def quicksort(numbers):",
    "num_assistant_tokens": 5
  }'| jq .
```
```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "text": "\n    if len(numbers) <= 1:\n        return numbers\n    else:\n        pivot = numbers[0]\n        lesser = [x for x in numbers[1:] if x <= pivot]\n        greater = [x for x in numbers[1:] if x > pivot]\n        return quicksort(lesser) + [pivot] + quicksort(greater)\n\n\ndef quicksort_recursive(numbers):\n    if"
    }
  ],
  "created": 1737547359,
  "model": "meta-llama/CodeLlama-7b-hf-sd",
  "object": "text_completion",
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 100,
    "total_tokens": 109
  }
}

```

High value for `num_assistant_tokens` brings profit when tokens generated by the draft model mostly match the main model. If they don't, tokens are dropped and both models do additional work. For low values such risk is lower, but the potential performance boost is limited. Usually the value of `5` is a good compromise.

Second speculative decoding specific parameter is `assistant_confidence_threshold ` which determines confidence level for continuing generation. If draft model generates token with confidence below that threshold, it stops generation for the current cycle and main model starts validation. `assistant_confidence_threshold` is a float in range (0, 1).

**Note that `num_assistant_tokens` and `assistant_confidence_threshold` are mutually exclusive.**