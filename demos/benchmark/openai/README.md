# OpenAI-Compatible Benchmarks {#ovms_demo_benchmark_openai}



## OVMS benchmark Script

The script benchmarks embeddings, reranking, audio, and VLM chat endpoints.
VLM mode uses streamed `/v1/chat/completions` requests and asks for the final
usage event with `stream_options.include_usage=true`.

```bash
git clone https://github.com/openvinotoolkit/model_server
cd model_server/demos/benchmark/openai
pip install -r requirements.txt
mkdir -p "${HOME}/models"
```

Each section starts one container named `ovms` on port 8000.
Run one use case at a time. After starting its server and before sending benchmark
requests, wait for the model to appear in `/v1/models`. Repeat the readiness check
in that section while the model is loading. Stop its server before starting the
next section.

All examples start with a small load. Increase `--limit` for longer measurements
and `--max_concurrency` for more concurrent clients. `--request_rate inf` sends
requests as quickly as the concurrency limit allows. The examples use explicit
model aliases; these can be omitted when `/v1/models` returns exactly one model.

| Use case | Source model | Script backend | Endpoint |
|---|---|---|---|
| VLM | `OpenVINO/LFM3-VL-2B-int4-ov` | `ovms-vlm` | `/v1/chat/completions` |
| Embeddings | `OpenVINO/Qwen3-Embedding-0.6B-int8-ov` | `ovms-embeddings` | `/v1/embeddings` |
| Reranking | `OpenVINO/bge-reranker-base-int8-ov` | `ovms_rerank` | `/v1/rerank` |
| Speech generation | `OpenVINO/Kokoro-82M-int8-ov` | `text2speech` | `/v1/audio/speech` |
| Speech transcription | `OpenVINO/whisper-large-v3-turbo-fp16-ov` | `speech2text` | `/v1/audio/transcriptions` |

## VLM Requests

Following the direct-download deployment in the
[VLM guide](../../continuous_batching/vlm/README.md), start LFM3-VL:

```bash
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run -d ${GPU_ARGS} --rm --name ovms \
	--user "$(id -u):$(id -g)" -p 8000:8000 \
	-v "${HOME}/models:/models:rw" \
	openvino/model_server:weekly \
	--rest_port 8000 --source_model OpenVINO/LFM3-VL-2B-int4-ov \
	--model_name vlm --model_repository_path /models
```

Wait for the model:

```bash
curl -f http://localhost:8000/v1/models
```

Download an image on the client:

```bash
curl -fL https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/static/images/bee.jpeg -o image.jpeg
```

The local image is sent as a data URI, so no server media-domain allowlist is
needed. For remote image URLs, configure `--allowed_media_domains` as described
in the VLM guide. OVMS provides the model tokenizer through
`/v1/tokenize`, so no Hugging Face tokenizer identifier is required:

```bash
python benchmark.py --backend ovms-vlm --model vlm --api_url http://localhost:8000/v1/chat/completions --input_length 128 --output_lenght 64 --max_concurrency 2 --limit 10 --image image.jpeg --turns 3 --turn_lenght 32
```

`--model` is optional. When omitted, the script queries `/v1/models` on the API
host and selects the model only if exactly one is listed. Multiple models produce
an error requiring `--model NAME`; empty or invalid model lists also fail.
An explicit `--model` skips discovery and accepts the served name or alias.
Backends without `/v1/models` require an explicit model.

`--image` accepts a local
filename, HTTP(S) URL, or image data URI. Local files are embedded as base64 data
URIs; remote image URLs must be accessible to the server.

Each independent VLM conversation starts with a random leading identifier in its
first user message. It is generated and token-checked before timing and counts
within `--input_length`. This prevents identical initial prompts from unintentionally
benefiting from cross-conversation prefix reuse. Prepared initial prompts are
checked for duplicates; if the token budget is too small to make them distinct,
the script fails and asks for a larger `--input_length`.

This does not disable the server's cache policy. Common chat-template prefixes or
other server-side caches may still be reused. Disable prefix caching in the model's
server configuration when a strict no-cache benchmark is required. For multi-turn
benchmarks, keep caching enabled: later turns retain the same initial identifier
and previous messages, then append the next user text and assistant response.

`--input_length` limits text tokens in the first user message, including the random
identifier but excluding image and chat-template tokens. To use text from a UTF-8
file instead of synthetic text:

```console
python benchmark.py --backend ovms-vlm --api_url http://localhost:8000/v1/chat/completions --text_input_file prompt.txt --input_length 128 --output_lenght 64 --max_concurrency 2 --limit 10 --image image.jpeg --turns 3 --turn_lenght 32
```

`--text_input_file` supplies the first user message for every conversation. The
script reads it once before timing, prepends a different random identifier per
conversation, and truncates the combined text to at most `--input_length` tokens
using the selected server or local tokenizer. File text is not padded or repeated,
but the identifier consumes part of the budget, leaving less space for file text.
Prefix truncation can end mid-word.
The option is supported only for `ovms-vlm`.

`--turn_lenght` adds new synthetic user text tokens on each later
turn. Previous user and assistant messages, including the initial image, are
resent as conversation history. Correctly spelled `--output_length` and
`--turn_length` aliases are also accepted.

`--output_lenght` sets `max_completion_tokens`; requests send `ignore_eos=true`.
`--limit` is the number of independent conversations and `--turns` includes the
first request. Turns within a conversation run sequentially; `--max_concurrency`
limits active conversations. VLM batch size is automatically 1. Failed turns stop
their conversation and produce a nonzero exit code.

Stop the server before starting the next use case:

```bash
docker stop ovms
```

## Text Embeddings

Serve a single pre-converted model directly, without `config.json` or a separate
pull step. Qwen3 embeddings use `LAST` pooling as described in the
[embeddings demo](../../embeddings/README.md):

```bash
docker run -d --rm --name ovms \
	--user "$(id -u):$(id -g)" -p 8000:8000 -v "${HOME}/models:/models:rw" \
	openvino/model_server:weekly --rest_port 8000 \
	--source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov \
	--model_repository_path /models
```

Wait for the model:

```bash
curl -f http://localhost:8000/v1/models
```

Benchmark synthetic text without downloading a dataset:

```bash
python benchmark.py --backend ovms-embeddings --api_url http://localhost:8000/v1/embeddings --dataset synthetic --synthetic_length 64 --batch_size 16 --limit 160 --max_concurrency 2 --request_rate inf
```

This sends 10 requests with 16 documents each. `--synthetic_length` repeats the
word `hi`; it is not an exact token count. The report counts the actual document
tokens through `/v1/tokenize` and reports tokens/second and mean/median latency.
To use real text instead, load a dataset such as Cohere Wikipedia:

```bash
python benchmark.py --backend ovms-embeddings --api_url http://localhost:8000/v1/embeddings --dataset Cohere/wikipedia-2023-11-embed-multilingual-v3 --hf-subset simple --batch_size 16 --limit 160 --max_concurrency 2 --request_rate inf
```

Stop the server before starting the next use case:

```bash
docker stop ovms
```

## Documents Reranking

Serve the pre-converted BGE model directly, without `config.json`, a separate
pull step, or PyTorch export. See the [rerank demo](../../rerank/README.md) for
the API and model details:

```bash
docker run -d --rm --name ovms \
	--user "$(id -u):$(id -g)" -p 8000:8000 \
	-v "${HOME}/models:/models:rw" \
	openvino/model_server:weekly --rest_port 8000 \
	--source_model OpenVINO/bge-reranker-base-int8-ov \
	--model_repository_path /models
```

Wait for the model:

```bash
curl -f http://localhost:8000/v1/models
```

Then run:

```bash
python benchmark.py --backend ovms_rerank --api_url http://localhost:8000/v1/rerank --dataset synthetic --synthetic_length 64 --batch_size 16 --limit 160 --max_concurrency 2 --request_rate inf
```

Each request ranks 16 documents against the script's fixed query `Hello`. Reported
tokens/second counts document tokens only, excluding that query and any server
template tokens. This is load generation, not a relevance-quality evaluation.
To benchmark real documents instead of synthetic text:

```bash
python benchmark.py --backend ovms_rerank --api_url http://localhost:8000/v1/rerank --dataset Cohere/wikipedia-2023-11-embed-multilingual-v3 --hf-subset simple --batch_size 16 --limit 160 --max_concurrency 2 --request_rate inf
```

Stop the server before starting the next use case:

```bash
docker stop ovms
```

## Speech Generation

Use the direct Kokoro deployment from the [audio demo](../../audio/README.md):

```bash
docker run -d --rm --name ovms \
	--user "$(id -u):$(id -g)" -p 8000:8000 \
	-v "${HOME}/models:/models:rw" \
	openvino/model_server:weekly --rest_port 8000 \
	--source_model OpenVINO/Kokoro-82M-int8-ov \
	--model_repository_path /models --target_device CPU
```

Wait for the model:

```bash
curl -f http://localhost:8000/v1/models
```

Use one text per request and a Kokoro voice:

```bash
python benchmark.py --backend text2speech --api_url http://localhost:8000/v1/audio/speech --dataset synthetic --synthetic_length 32 --batch_size 1 --limit 10 --max_concurrency 2 --request_rate inf --voice af_alloy
```

The backend is spelled `text2speech`. The voice `af_alloy` corresponds to a file in
the model's `voices/` directory. The script decodes the returned WAV to report mean
and median RTFx (`generated audio seconds / request latency`), plus success rate
and request latency. RTFx above 1 means faster-than-real-time generation.
For a real text corpus, for example AMI's individual head microphone transcripts,
run:

```bash
python benchmark.py --backend text2speech --api_url http://localhost:8000/v1/audio/speech --dataset edinburghcstr/ami --hf-subset ihm --batch_size 1 --limit 10 --max_concurrency 2 --request_rate inf --voice af_alloy
```

Stop the server before starting the next use case:

```bash
docker stop ovms
```

## Speech Transcription

Start the pre-converted Whisper model using the direct-download pattern from the
[audio demo](../../audio/README.md):

```bash
docker run -d --rm --name ovms \
	--user "$(id -u):$(id -g)" -p 8000:8000 \
	-v "${HOME}/models:/models:rw" \
	openvino/model_server:weekly --rest_port 8000 \
	--source_model OpenVINO/whisper-large-v3-turbo-fp16-ov \
	--model_repository_path /models
```

The current audio request code expects decoded dataset records with `array` and
`sampling_rate` fields. Use Datasets 3.x for this array-based loader; newer
Datasets versions can return AudioDecoder objects instead. In the benchmark
virtual environment, after the common requirements installation:

```bash
python -m pip install "datasets[audio]==3.6.0"
```

Wait for the model:

```bash
curl -f http://localhost:8000/v1/models
```

Then use a small real-audio dataset:

```bash
python benchmark.py --backend speech2text --api_url http://localhost:8000/v1/audio/transcriptions --dataset hf-internal-testing/librispeech_asr_dummy --hf-subset clean --split validation --batch_size 1 --limit 10 --max_concurrency 2 --request_rate inf --tokenizer openai/whisper-large-v3-turbo
```

This downloads audio clips from Hugging Face and uploads WAV data as multipart
requests. A synthetic text dataset is not valid for transcription. The tokenizer
override is needed because this backend locally counts tokens in the returned
transcripts, while the served alias `whisper` is not a Hugging Face tokenizer
identifier. The report includes success rate, transcript tokens/second, and
successful-request mean/median latency.

Stop the container when finished:

```bash
docker stop ovms
```

## Timing and Usage

The report includes mean, median, p90, p99, and standard deviation of successful
request latency, plus separate mean and median timing metrics:

| Metric | Definition |
|---|---|
| TTFT | Request start to first non-empty generated-text event, including reasoning text when supplied. Role-only and empty events are ignored. |
| TPOT | Time from first to last generated-text event divided by `usage.completion_tokens` from the final usage event. A single text event yields zero elapsed time; zero output tokens or no text events produce no TPOT sample. |
| Inter-chunk latency | Gaps between successive non-empty generated-text events, pooled across successful requests. |

TPOT mean/median aggregates per-request values using the server-reported output
token count, not an inferred count of chunks or locally tokenized text.
Inter-chunk mean/median aggregates individual arrival gaps. SSE events can contain
multiple tokens, and network buffering can group events, so the timestamps are
client-observed arrivals, not model-internal decode-step times. Inter-chunk gaps equal inter-token gaps only
when each event represents one token. The final usage event and `[DONE]` do not
extend TPOT. Overall request latency includes reading the complete stream.

Token statistics come from `usage.prompt_tokens`, `usage.completion_tokens`, and
`usage.prompt_tokens_details.cached_tokens`. Cached tokens are already included
in prompt tokens: uncached input equals prompt minus cached. The report includes
these totals, cache-hit rate, input/output token throughput, request throughput,
and attempted versus planned requests. Failed requests are excluded from token
and timing statistics. Missing usage or an incomplete stream fails the request.


## vLLM bench tool



Install vLLM and download the Spec-Bench
```bash
pip install vllm[bench] --extra-index-url https://wheels.vllm.ai/0.32.0/cpu --extra-index-url https://download.pytorch.org/whl/cpu
curl -O https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
```

Run benchmark with 10 requests sent sequentially:
```bash
vllm bench serve --model OpenVINO/Qwen3.8-27B-int4-ov --dataset-name spec_bench --dataset-path "question.jsonl" --num-prompts 10 \
--spec-bench-category "summarization" --backend openai-chat --host localhost --port 8000 --endpoint /v1/chat/completions --max-concurrency 1
```

Refer to [vLLM bench documentation](https://docs.vllm.ai/en/latest/benchmarking/cli) to learn about other parameters



## aiperf

```bash
pip install "aiperf==0.13.0"
aiperf profile --help
```

Deploy the VLM from the section above and wait for `/v1/models`. AIPerf requires
the served model name explicitly: use `vlm` for that deployment, or substitute the
ID returned by `/v1/models`. Unlike the Python script, `--url` is the server root,
without `/v1`; `--endpoint-type chat` appends `/v1/chat/completions`.

### Image and Streaming Smoke Test

Send two sequential requests, each containing one generated 224x224 JPEG image:

```bash
aiperf profile --model vlm --url http://localhost:8000 \
	--endpoint-type chat --streaming --tokenizer builtin \
	--concurrency 1 --request-count 2 \
	--prompt-input-tokens-mean 32 --prompt-input-tokens-stddev 0 \
	--output-tokens-mean 16 --output-tokens-stddev 0 \
	--image-batch-size 1 --image-width-mean 224 --image-height-mean 224 \
	--image-format jpeg --extra-inputs ignore_eos:true \
	--request-timeout-seconds 120 --no-server-metrics --no-gpu-telemetry \
	--ui simple --artifact-dir ./artifacts/aiperf-vlm-smoke
```

Streaming enables time-to-first-token and inter-token timing. Output length is
passed as `max_completion_tokens`; `ignore_eos:true` asks OVMS not to stop early
on EOS. `--tokenizer builtin` avoids downloading a model tokenizer, but its prompt
token lengths are not exact counts for the served VLM. Image and chat-template
tokens also increase the server-reported prompt usage.

### Single-Client Latency

Use a longer run at concurrency 1, with a larger text prompt:

```bash
aiperf profile --model vlm --url http://localhost:8000 \
	--endpoint-type chat --streaming --tokenizer builtin \
	--concurrency 1 --request-count 20 \
	--prompt-input-tokens-mean 128 --prompt-input-tokens-stddev 0 \
	--output-tokens-mean 64 --output-tokens-stddev 0 \
	--image-batch-size 1 --image-width-mean 224 --image-height-mean 224 \
	--image-format jpeg --extra-inputs ignore_eos:true \
	--request-timeout-seconds 120 --no-server-metrics --no-gpu-telemetry \
	--ui simple --artifact-dir ./artifacts/aiperf-vlm-latency
```

### Concurrent Throughput

Keep the same input/output and image settings, but increase concurrency:

```bash
aiperf profile --model vlm --url http://localhost:8000 \
	--endpoint-type chat --streaming --tokenizer builtin \
	--concurrency 20 --request-count 100 \
	--prompt-input-tokens-mean 128 --prompt-input-tokens-stddev 0 \
	--output-tokens-mean 64 --output-tokens-stddev 0 \
	--image-batch-size 1 --image-width-mean 224 --image-height-mean 224 \
	--image-format jpeg --extra-inputs ignore_eos:true \
	--request-timeout-seconds 120 --no-server-metrics --no-gpu-telemetry \
	--ui simple --artifact-dir ./artifacts/aiperf-vlm-throughput
```

For natural images instead of generated noise, add
`--image-source /path/to/image-directory`; AIPerf samples and resizes those images
to the selected dimensions. The generated-image examples require no image file
or server-side URL access. For larger images, change both width and height and
compare results with otherwise unchanged parameters.

