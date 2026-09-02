# LLM Models in Speculative Decoding Pipeline{#ovms_demos_continuous_batching_speculative_decoding}

Speculative (assisted) decoding reduces generation latency without changing the output distribution. A lightweight drafter proposes candidate tokens; the main model validates them in one parallel forward pass. Accepted draft tokens replace sequential decode steps of the main model, yielding end-to-end speedups that are most pronounced at concurrency 1.

OpenVINO GenAI implements three drafting strategies, all exposed through the same `draft_models_path` configuration field in OVMS:

| Strategy | How it drafts | Best for | Extra model required |
|---|---|---|---|
| **Fast Draft** | Small off-the-shelf LLM | General-purpose; any target/draft pair | Yes — smaller LLM sharing target's tokenizer |
| **EAGLE3** | Draft head conditioned on target's hidden states | Highest acceptance rate; code and reasoning; supports tree drafting | Yes — EAGLE3 head trained on the target family |
| **MTP** | Built-in multi-token prediction head | Models with bundled MTP heads (e.g. Qwen3.8-27B) | No — head bundled with the main model |

All three strategies share the same server API — only the generation parameters differ.

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

# EAGLE3

EAGLE3 replaces the generic draft model with a small head — typically one transformer layer — trained to predict the next token from the target model's hidden states. Because it sees the same internal representation as the target, its acceptance rate is substantially higher than Fast Draft on the same target.

EAGLE3 supports two candidate generation modes:

- **Chain drafting** (default) — runs the draft head autoregressively for `num_assistant_tokens` steps and submits a linear chain of candidates.
- **Tree drafting** — expands `branching_factor` top-k continuations at each of `tree_depth` layers, then submits the highest-scoring `num_assistant_tokens` candidates in one packed verification step. This compounds the already-high EAGLE3 acceptance rate into longer accepted runs per target step, at the cost of a larger validation batch. Best on GPU at small batch sizes.

## Model considerations

For this demo we use a model pair from [available EAGLE3 models](https://github.com/SafeAILab/EAGLE#eagle-3-models-on-hugging-face):
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as a main model
- [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3) as a draft model

both in INT4 precision.

## Model preparation

Python environment setup:
```console
# Install regular requirements for OVMS export script
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt

mkdir models
```

Run `export_model.py` script to download and quantize the model:

```console
python export_model.py text_generation --source_model Qwen/Qwen3-8B --draft_source_model AngelSlim/Qwen3-8B_eagle3 --draft_eagle3_mode --weight-format int4 --config_file_path models/config.json --model_repository_path models
```

Draft model inherits all scheduler properties from the main model.

You should have a model folder like below:
```
models
├── config.json
└── Qwen
    └── Qwen3-8B
        ├── added_tokens.json
        ├── AngelSlim-Qwen3-8B_eagle3
        │   ├── config.json
        │   ├── generation_config.json
        │   ├── openvino_config.json
        │   ├── openvino_model.bin
        │   └── openvino_model.xml
        ├── chat_template.jinja
        ├── config.json
        ├── generation_config.json
        ├── graph.pbtxt
        ├── merges.txt
        ├── openvino_config.json
        ├── openvino_detokenizer.bin
        ├── openvino_detokenizer.xml
        ├── openvino_model.bin
        ├── openvino_model.xml
        ├── openvino_tokenizer.bin
        ├── openvino_tokenizer.xml
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.json

```

## Server Deployment

:::{dropdown} **Deploying with Docker**
```bash
docker run -d --rm $(test -d /dev/dri && echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)") \
  -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:weekly \
  --rest_port 8000 --config_path /workspace/config.json
```

OVMS auto-detects the best available device at startup. To target a specific device explicitly, pass `--target_device GPU` (or `NPU`, `HETERO:GPU,CPU`, etc.) to `export_model.py` and follow the [AI accelerators guide](../../../docs/accelerators.md) for additionally required docker parameters.
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --config_path ./models/config.json
```
:::

## Check performance

Let's check how the deployed model is doing by running performance test. For that purpose we can use vLLM benchmark script and sonnet dataset.

Install vLLM and download sonnet dataset: 
```bash
pip install vllm --index-url https://wheels.vllm.ai/nightly/cpu --extra-index-url https://pypi.org/simple
curl https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/benchmarks/sonnet.txt -o sonnet.txt
```

Run benchmark with 100 requests sent sequentially:
```bash
vllm bench serve --dataset-name sonnet --dataset-path sonnet.txt --backend openai-chat --host localhost --port 8000 --endpoint /v3/chat/completions --max-concurrency 1 --tokenizer Qwen/Qwen3-8B --model Qwen/Qwen3-8B --num_prompts 100

Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [06:59<00:00,  4.19s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  419.00
Total input tokens:                      54256
Total generated tokens:                  15000
Request throughput (req/s):              0.24
Output token throughput (tok/s):         35.80
Peak output token throughput (tok/s):    16.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):          165.29
---------------Time to First Token----------------
Mean TTFT (ms):                          426.71
Median TTFT (ms):                        424.97
P99 TTFT (ms):                           635.37
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.25
Median TPOT (ms):                        25.09
P99 TPOT (ms):                           29.22
---------------Inter-token Latency----------------
Mean ITL (ms):                           66.29
Median ITL (ms):                         66.75
P99 ITL (ms):                            72.11
==================================================
```

## Chain drafting

Send `num_assistant_tokens` to control how many candidates the draft head proposes per target step:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "What is OpenVINO?"}],
    temperature=0,
    max_tokens=200,
    extra_body={"num_assistant_tokens": 5},
)
print(response.choices[0].message.content)
```

Increase `num_assistant_tokens` until the tokens-per-step figure plateaus, then back off — past the plateau, rejected draft tokens are pure overhead.

Setting `num_assistant_tokens: 0` disables drafting for that request; only the target model runs.

## Tree drafting

Tree drafting adds two `GenerationConfig` fields. Setting `tree_depth > 0` switches from chain to tree mode:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "What is OpenVINO?"}],
    temperature=0,
    max_tokens=200,
    extra_body={
        "num_assistant_tokens": 15,   # candidates verified per step
        "branching_factor": 8,        # top-k expansions per tree layer
        "tree_depth": 4,              # draft head iterations
    },
)
```

`total_draft_tokens = branching_factor² × (tree_depth − 1) + branching_factor` must be ≥ `num_assistant_tokens`. A reasonable starting point is `branching_factor=4..8`, `tree_depth=3..4`.

Tree drafting is EAGLE3-only; it cannot be combined with beam search or multinomial sampling.

# Fast Draft

Fast Draft is the classic two-model setup: a smaller off-the-shelf LLM that shares the target's tokenizer proposes tokens autoregressively, and the target model verifies them. It works with any target/draft pair without retraining. The speedup depends on how often the small model's distribution agrees with the large one.

## Model considerations

Both models must share the same tokenizer so draft token IDs map correctly to target token IDs.

Performance gain depends heavily on the model pair and workload — the optimal combination should be found empirically. Model sizes and precisions both factor in.

In this demo:
  - [meta-llama/CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf) as a main model
  - [AMD-Llama-135m](https://huggingface.co/amd/AMD-Llama-135m) as a draft model

both in FP16 precision.

## Model preparation
Here, the original Pytorch LLM models and the tokenizers will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Download export script, install its dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
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

OVMS auto-detects the best available device at startup. To target a specific device explicitly, pass `--target_device GPU` (or `NPU`, `HETERO:GPU,CPU`, etc.) to `export_model.py` and follow the [AI accelerators guide](../../../docs/accelerators.md) for additionally required docker parameters.
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

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

Models used in this demo — `meta-llama/CodeLlama-7b-hf` and `AMD-Llama-135m` — are base (non-chat) models, so we use the `completions` endpoint.

```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.completions.create(
    model="meta-llama/CodeLlama-7b-hf",
    prompt="<s>def quicksort(numbers):",
    temperature=0,
    max_tokens=100,
    extra_body={"num_assistant_tokens": 5},
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="", flush=True)
```

**`num_assistant_tokens`** controls how many tokens the draft model proposes before the main model validates them. High values pay off when the draft frequently agrees with the target; low values reduce wasted work when it doesn't. `5` is a good starting point.

**`assistant_confidence_threshold`** is an alternative stopping criterion: the draft keeps proposing while its token probability exceeds the threshold, then hands off to the target. It is mutually exclusive with `num_assistant_tokens`. Supported on the Continuous Batching backend only (`LM_CB`) — the stateful backend (`pipeline_type: LM`) does not implement dynamic-length drafting.

`num_assistant_tokens` does not have to be sent on every request — see [Setting default generation parameters](#setting-default-generation-parameters) to configure a deployment-level default.

# MTP (Multi-Token Prediction)

MTP replaces the separate draft model with a lightweight prediction head bundled inside the main model weights — no additional download is needed. The head is auto-detected by OVMS when `openvino_mtp_model.xml` is present in the draft model directory. Because it shares the main model's weights, the draft cost is minimal and acceptance rates are high for the same model family.

## Model considerations

For this demo we use [OpenVINO/Qwen3.8-27B-int8-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int8-ov) — a vision-language model with a bundled MTP head exported in INT8 precision.

> **Note:** This model requires OpenVINO nightly builds and is marked experimental. See the model card for compatibility details.
> **Note:** Prefix caching is not yet supported in this mode.

## Server Deployment

:::{dropdown} **Deploying with Docker**
```bash
docker run -d --rm $(test -d /dev/dri && echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)") \
  -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly \
  --rest_port 8000 \
  --model_repository_path /models \
  --source_model OpenVINO/Qwen3.8-27B-int4-ov \
  --draft_model_path . \
  --enable_prefix_caching false
```
:::

:::{dropdown} **Deploying on Bare Metal**
```console
ovms --rest_port 8000 \
  --model_repository_path c:\models \
  --source_model OpenVINO/Qwen3.8-27B-int4-ov \
  --draft_model_path . \
  --enable_prefix_caching false
```
:::

## Request Generation

The API is identical to other speculative decoding strategies:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="OpenVINO/Qwen3.8-27B-int4-ov",
    messages=[{"role": "user", "content": "Explain the transformer attention mechanism."}],
    temperature=0,
    max_tokens=200,
    extra_body={"num_assistant_tokens": 5},
)
print(response.choices[0].message.content)
```

`num_assistant_tokens` controls how many MTP candidates are proposed per target step. The default is `5` if not specified.

# Setting Default Generation Parameters

The main model's `generation_config.json` is read at server start-up as the default generation configuration for all requests. Parameters absent from the request body fall back to this file, then to OVMS built-in defaults.

**Resolution order: request body → `generation_config.json` → OVMS built-in default**

To set a deployment-level default for any assisted decoding parameter, edit `generation_config.json` in the main model directory:

```json
{
    "num_assistant_tokens": 7
}
```

The built-in fallback for `num_assistant_tokens` is `5`. All other generation parameters (`temperature`, `max_new_tokens`, `top_p`, etc.) follow the same resolution order.
