# LLM Models in Speculative Decoding Pipeline{#ovms_demos_continuous_batching_speculative_decoding}

Speculative (assisted) decoding reduces generation latency without changing the output distribution. A lightweight drafter proposes candidate tokens; the main model validates them in one parallel forward pass. Accepted draft tokens replace sequential decode steps of the main model, yielding end-to-end speedups that are most pronounced at concurrency 1.

OpenVINO GenAI implements three drafting strategies, all exposed through the same `draft_models_path` configuration field in OVMS:

| Strategy | How it drafts | Best for | Extra model required |
|---|---|---|---|
| **MTP** | Built-in multi-token prediction head | Models with bundled MTP heads (e.g. Qwen3.8-27B) | No — head bundled with the main model |
| **Fast Draft** | Small off-the-shelf LLM | General-purpose; any target/draft pair | Yes — smaller LLM sharing target's tokenizer |
| **EAGLE3** | Draft head conditioned on target's hidden states | Highest acceptance rate; code and reasoning; supports tree drafting | Yes — EAGLE3 head trained on the target family |

All three strategies share the same server API — only the generation parameters differ.

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip and a Hugging Face account

**Model Server deployment**: Docker Engine or the OVMS binary package installed according to the [bare-metal deployment guide](../../../docs/deploying_server_baremetal.md)

# MTP (Multi-Token Prediction)

MTP replaces the separate draft model with a lightweight prediction head bundled inside the main model weights — no additional download is needed. The head is auto-detected by OVMS when `openvino_mtp_model.xml` is present in the draft model directory. Because it shares the main model's weights, the draft cost is minimal and acceptance rates are high for the same model family.

## Model considerations

For this demo we use [OpenVINO/Qwen3.8-27B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov), which has a bundled MTP head and is exported in INT4 precision.

> **Note:** This model requires OVMS 2026.4 or weekly pre-release build. See the model card for compatibility details. Prefix caching is not currently supported in MTP mode. It is also required to set max_tokens parameter

## Server Deployment

:::{dropdown} **Deploying with Docker**
```bash
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run -d --rm ${GPU_ARGS} --user $(id -u):$(id -g) -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly \
  --rest_port 8000 \
  --model_repository_path /models \
  --source_model OpenVINO/Qwen3.8-27B-int4-ov \
  --draft_model_path . \
  --enable_prefix_caching false
```
:::

:::{dropdown} **Deploying on Bare Metal**
```bat
ovms --rest_port 8000 --model_repository_path c:\models --source_model OpenVINO/Qwen3.8-27B-int4-ov --draft_model_path . --enable_prefix_caching false
```
:::

## Request Generation

The API is identical to other speculative decoding strategies:
```console
pip install openai
```
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="OpenVINO/Qwen3.8-27B-int4-ov",
    messages=[{"role": "user", "content": "Explain the transformer attention mechanism."}],
    temperature=0,
    extra_body={"num_assistant_tokens": 5},
)
print(response.choices[0].message)
```

`num_assistant_tokens` controls how many MTP candidates are proposed per target step. The default is `5` if not specified.

## Check performance

Check the deployed model's performance by using the vLLM benchmark script and the Sonnet dataset.

Install vLLM and download the Sonnet dataset:
```bash
pip install vllm --index-url https://wheels.vllm.ai/nightly/cpu --extra-index-url https://pypi.org/simple
curl https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/benchmarks/sonnet.txt -o sonnet.txt
```

Run benchmark with 100 requests sent sequentially:
```bash
vllm bench serve --dataset-name sonnet --dataset-path sonnet.txt --backend openai-chat --host localhost --port 8000 --endpoint /v1/chat/completions --max-concurrency 1 --model OpenVINO/Qwen3.8-27B-int4-ov --num-prompts 10
```
```
============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  27.85
Total input tokens:                      5405
Total generated tokens:                  1500
Request throughput (req/s):              0.36
Output token throughput (tok/s):         53.86
Peak output token throughput (tok/s):    72.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):          247.95
---------------Time to First Token----------------
Mean TTFT (ms):                          496.63
Median TTFT (ms):                        433.09
P99 TTFT (ms):                           897.54
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.35
Median TPOT (ms):                        15.15
P99 TPOT (ms):                           17.88
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.02
Median ITL (ms):                         0.02
P99 ITL (ms):                            54.90
==================================================
```


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

```bat
python export_model.py text_generation --source_model Qwen/Qwen3-8B --draft_source_model AngelSlim/Qwen3-8B_eagle3 --draft_eagle3_mode --weight-format int4 --model_repository_path c:\models
```
or
```bash
python export_model.py text_generation --source_model Qwen/Qwen3-8B --draft_source_model AngelSlim/Qwen3-8B_eagle3 --draft_eagle3_mode --weight-format int4 --model_repository_path ${HOME}/models
```

Draft model inherits all scheduler properties from the main model.

You should have a model folder like below:
```
models
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
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run -d ${GPU_ARGS} --user $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:ro openvino/model_server:weekly \
    --model_path /models/Qwen/Qwen3-8B \
    --model_name Qwen/Qwen3-8B \
    --rest_port 8000
```

:::

:::{dropdown} **Deploying on Bare Metal**

Install OVMS as described in the [deployment guide](../../../docs/deploying_server_baremetal.md).

```bat
ovms --rest_port 8000 --model_path c:\models\Qwen\Qwen3-8B --model_name Qwen/Qwen3-8B
```
:::


## Chain drafting

Send `num_assistant_tokens` to control how many candidates the draft head proposes per target step:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "What is OpenVINO?"}],
    temperature=0,
    max_tokens=2000,
    extra_body={"num_assistant_tokens": 5},
)
print(response.choices[0].message.content)
```

Increase `num_assistant_tokens` until the tokens-per-step figure plateaus, then back off — past the plateau, rejected draft tokens are pure overhead.

Setting `num_assistant_tokens: 0` disables drafting for that request; only the target model runs.

## Tree drafting

Tree drafting adds two `GenerationConfig` fields. Setting `tree_depth > 0` switches from chain to tree mode:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "What is OpenVINO?"}],
    temperature=0,
    max_tokens=2000,
    extra_body={
        "num_assistant_tokens": 5,    # candidates verified per step
        "branching_factor": 4,        # top-k expansions per tree layer
        "tree_depth": 2,              # draft head iterations
    },
)
print(response.choices[0].message.content)
```

`total_draft_tokens = branching_factor² × (tree_depth − 1) + branching_factor` must be ≥ `num_assistant_tokens`. A reasonable starting point is `branching_factor=4..8`, `tree_depth=3..4`.

Tree drafting is EAGLE3-only; it cannot be combined with beam search or multinomial sampling.



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
