# GGUF Frontend Enablement Plan

## Goal

Use the OpenVINO GGUF frontend by default when OVMS serves a `.gguf` model, with no new command-line or model-config parameter. Keep existing non-GGUF model loading unchanged and retain an explicit `GGUF_READER` override for diagnosis or fallback.

## Current Integration

- OVMS accepts GGUF paths in `GenAiServableInitializer` and constructs either `ov::genai::LLMPipeline` or `ov::genai::ContinuousBatchingPipeline`.
- Both construction paths already consume `plugin_config`, which can carry GenAI properties.
- GenAI exposes `ov::genai::gguf_reader("FRONTEND")`; its legacy reader remains the default in the package currently selected by `versions.mk` (2026.5.0.0rc1).
- The referenced GenAI test suite compares tiny GGUF fixtures with Optimum references and separately generates synthetic GGUFs for architecture coverage.

## GenAI Test Model Coverage

The following are the pinned real GGUF fixtures in [`test_gguf_frontend.py`](https://github.com/openvinotoolkit/openvino.genai/blob/79168ddb0542b8e2ac196cf517a7f385de8b2f1e/tests/python_tests/test_gguf_frontend.py). Each source tokenizer/configuration is pinned independently; the reference model is reconstructed from that fixture's GGUF weights.

| Test ID | GGUF repository @ revision | GGUF file | Source tokenizer/config @ revision |
| --- | --- | --- | --- |
| llama | `aladar/llama-2-tiny-random-GGUF` @ `8d5321916486e1d33c46b16990e8da6567785769` | `llama-2-tiny-random.gguf` | `yujiepan/llama-2-tiny-random` @ `74bb065d6381fdf9fcde5896cb32267ba2c0ee0a` |
| qwen3 | `mradermacher/tiny-random-qwen3-GGUF` @ `058102830a84bb976597f4d8462bafe0ea319358` | `tiny-random-qwen3.f16.gguf` | `snake7gun/tiny-random-qwen3` @ `add6406ed2f82988fd5194e343572194f71b6f61` |
| phi3 | `mradermacher/tiny-random-Phi3ForCausalLM-GGUF` @ `ff12230de1587e824688cc6afef2af22dc2b7807` | `tiny-random-Phi3ForCausalLM.f16.gguf` | `Xenova/tiny-random-Phi3ForCausalLM` @ `bac53a980e9402513e0a5f8433656b4f845c1b9b` |
| gemma3 | `ggml-org/tinygemma3-GGUF` @ `c287502cd9e278dac8eed805c112cce5d0081e0b` | `tinygemma3-Q8_0.gguf` | `ngxson/tinygemma3_cifar` @ `da67f59d2195a0095711f1494f8ee5823cf54591` |

The separate generated-model suite, [`test_gguf_generated.py`](https://github.com/openvinotoolkit/openvino.genai/blob/79168ddb0542b8e2ac196cf517a7f385de8b2f1e/tests/python_tests/test_gguf_generated.py), generates these 23 GGUF variants. The filenames follow the test's `architecture-variant.gguf` convention.

| Generated GGUF fixture | Generated GGUF fixture |
| --- | --- |
| `bailingmoe2-moe.gguf` | `ernie4_5-moe-moe.gguf` |
| `exaone4-dense.gguf` | `gemma-dense.gguf` |
| `gemma2-dense.gguf` | `gpt-oss-moe.gguf` |
| `hunyuan-dense-dense.gguf` | `llama-dense.gguf` |
| `llama-moe.gguf` | `maincoder-dense.gguf` |
| `mamba2-dense.gguf` | `minicpm-dense.gguf` |
| `minicpm-moe.gguf` | `mistral3-dense.gguf` |
| `mistral3-moe.gguf` | `nemotron_h-dense.gguf` |
| `olmoe-moe.gguf` | `phi3-dense.gguf` |
| `qwen2-dense.gguf` | `qwen3-dense.gguf` |
| `qwen35-dense.gguf` | `qwen3moe-moe.gguf` |
| `smollm3-dense.gguf` | |

These are synthetic two-layer weights with numeric-token inputs, not production checkpoints. This is architecture coverage, not a guarantee for every real model or quantization.

## Benchmark Pair Proposals

These pairs align to models already used by the OVMS validation/performance team. The OpenVINO IR and GGUF artifacts should come from the same source checkpoint and tokenizer. For production-format performance tests, retain each artifact's native quantization; interpret accuracy differences with quantization differences in mind.

| Priority | OpenVINO model / source checkpoint | GGUF model | Notes |
| --- | --- | --- | --- |
| 1 | `OpenVINO/Qwen3-8B-int4-ov` (source: `Qwen/Qwen3-8B`) | `Qwen/Qwen3-8B-GGUF` (use a Q4_K_M or Q8_0 file) | Best first production-scale pair. The OVMS accuracy matrix includes Qwen3 models; GenAI's real-checkpoint GGUF accuracy case is Qwen3-0.6B, so 8B is additional scale coverage. |
| 2 | `OpenVINO/gpt-oss-20b-int4-ov` (source: `openai/gpt-oss-20b`) | `ggml-org/gpt-oss-20b-GGUF` (`gpt-oss-20b-mxfp4.gguf`) | Same source checkpoint; included in GenAI's real-GGUF accuracy suite. Heavier than Qwen3-8B and uses a different quantization format across artifacts. |
| 3 | OpenVINO IR exported from `meta-llama/Meta-Llama-3.1-8B-Instruct` | `unsloth/Llama-3.1-8B-Instruct-GGUF` (choose a Q4_K_M or Q8_0 file) | The full 8B IR is the comparison target. `OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov` is only a speculative draft model, not the full-model counterpart. The source checkpoint requires Hugging Face access approval. |

For strict accuracy comparisons, use reference weights from the same source checkpoint and record each conversion's precision/quantization. For performance, compare the practical production artifacts with identical prompts, generation settings, device, and concurrency; report cold load, TTFT, decode latency, throughput, and memory separately.

## OVMS Launch Examples

These commands were run inside the configured `-build` container from `/ovms` after building `//src:ovms`. The first starts the pinned test model through OVMS's Hugging Face pull path; `GGUF_READER` is intentionally omitted.

```text
./bazel-bin/src/ovms \
	--rest_port 8097 \
	--model_repository_path /tmp/ovms_gguf_frontend \
	--task text_generation \
	--source_model mradermacher/tiny-random-Phi3ForCausalLM-GGUF \
	--gguf_filename tiny-random-Phi3ForCausalLM.f16.gguf \
	--model_name tiny-phi3
```

The second instance used the GenAI test's reference precision settings for the accuracy comparison. These `plugin_config` values were test-only; they are not needed to select the GGUF frontend.

```text
./bazel-bin/src/ovms \
	--rest_port 8098 \
	--model_repository_path /tmp/ovms_gguf_frontend_ref \
	--task text_generation \
	--source_model mradermacher/tiny-random-Phi3ForCausalLM-GGUF \
	--gguf_filename tiny-random-Phi3ForCausalLM.f16.gguf \
	--model_name tiny-phi3-ref \
	--plugin_config '{"INFERENCE_PRECISION_HINT":"f32","KV_CACHE_PRECISION":"f16","DYNAMIC_QUANTIZATION_GROUP_SIZE":0}'
```

Smoke request sent to the default instance:

```text
curl --noproxy '*' -sS -H 'Content-Type: application/json' \
	-d '{"model":"tiny-phi3","prompt":"Why is the Sun yellow?","max_tokens":16,"temperature":0,"stream":false}' \
	http://127.0.0.1:8097/v3/completions
```

## Implementation

1. In the shared GenAI initializer path, detect that the resolved model path is a `.gguf` file.
2. Set `GGUF_READER` to `FRONTEND` only when it was not explicitly provided in `plugin_config`. Apply this consistently before constructing both legacy and continuous-batching pipelines; leave IR directories and all non-GGUF paths unchanged.
3. Add focused unit coverage for default injection, explicit override preservation, and non-GGUF behavior. Add or adapt an opt-in end-to-end GGUF test that loads the pinned tiny Phi-3 model through OVMS and confirms the server is ready and produces a completion.
4. Update GGUF deployment documentation to state that OVMS selects the frontend by default and document the override only as a troubleshooting escape hatch.

## Validation: Accuracy and Performance

- Use the pinned Phi-3 GGUF and matching tokenizer/config revisions from GenAI's `test_gguf_frontend.py`; do not commit model files. Cache downloads only in a local test cache.
- Accuracy: compare OVMS tokenization/completion against GenAI's Optimum reference path using fixed prompts and deterministic greedy generation (fixed token budget, no sampling). Record exact token-ID agreement where the OVMS API exposes it; otherwise compare decoded output and report the limitation. Also confirm the model loads without reader-specific workarounds.
- Performance: measure cold initialization separately from warmed inference. On the same host/device and request settings, collect repeated prompt-prefill and decode latency plus throughput for the frontend and legacy reader where the selected architecture supports both. Report median and p95 with hardware, thread/device settings, package version, and cache state. For Phi-3, frontend-only support is the primary pass criterion if the legacy path does not support that architecture.
- First run the narrow OVMS test and the corresponding GenAI frontend test; then run the benchmark repetitions. Keep performance results informational until a stable OVMS baseline and acceptance threshold are agreed.

## Risks and Acceptance

- The frontend is a preview feature; architecture, tokenizer, quantization, and attention-backend support vary by model. A failed model must not silently fall back to legacy parsing.
- Verify that both GenAI pipeline constructors honor the same reader property and that an explicit user override remains effective.
- Accept when a `.gguf` request works without adding `GGUF_READER`, non-GGUF loading is unaffected, the pinned fixture matches the reference under deterministic settings, and measured performance/accuracy results are recorded with enough environment detail to reproduce them.

## Validation Results
- OVMS loaded the pinned tiny Phi-3 GGUF without setting `GGUF_READER`. With the GenAI reference settings (`INFERENCE_PRECISION_HINT=f32`, `KV_CACHE_PRECISION=f16`, `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`), OVMS exactly matched the independent Transformers/PyTorch reconstruction for 16 greedy tokens. Weight loading reported zero missing, unexpected, or mismatched tensors. The prompt token count was 7.
- The default OVMS precision policy also loaded and generated deterministically, but produced different tokens than the F32 reference on this randomly initialized fixture. This separates reader selection from precision policy; no precision default was changed.
- Warmed sequential CPU measurements, 20 requests each, 7 prompt tokens and 16 generated tokens: default settings had 94.8 ms median / 105.2 ms p95 (about 169 tokens/s by request wall time); reference precision settings had 23.2 ms median / 30.2 ms p95 (about 661 tokens/s). No legacy-reader comparison was made because Phi-3 is outside its documented architecture support.
- Accuracy reference used Transformers reconstruction from the GGUF and pinned source tokenizer/config revisions, not the GenAI suite's Optimum-exported IR. Performance numbers are local smoke measurements, not a production baseline.