# Baseline models and available speculative drafters

**Status:** prioritization correction and source-supported future acceptance plan (21 Oct 2026). **No runtime changes.**

## Primary model tiers

1. **Gemma 4 26B-A4B / GEMMAMONSTER C6**: **golden protocol baseline**. The success criterion is complete and robust tool-call semantics, including reasoning/tool transitions, parallel calls, several OVMS slots and multi-turn loops. Gemma is not a proxy for the coding-agent performance of Qwen3.6. Keep model, quantization, tokenizer, prompt overlay and accepted C6 runtime pinned. This was sometimes called “Gemma3-26B A4B” in chat; the intended model family is **Gemma 4 26B-A4B**.
2. **Qwen3.6-35B-A3B**: **top priority for production-style agentic OTD performance**. Development, tool use, long context, batch concurrency, KV U4, and speculative verification must all be represented.
3. **AEON Ornith-1.0-35B**: **second production-style agentic target**. It is a separate fine-tuned / modified checkpoint with potentially different expert activation distribution. Do not assume equal working set or drafter acceptance just because it shares a model-family architecture.
4. **Qwen3.8-Flash-Next**: **high-value experimental reference**: built-in MTP, verify windows, different expert quantizations by layer and matrix role, large-model NVMe behavior. Do not require it to fit 32 GB Arc 140V before applying the design insights.
5. **Qwen3-30B-A3B**: legacy / differential OTD control to test a known kernel path, **not** what defines the product acceptance gates.

## Draft candidates and present evidence

| Target | Drafter and artifact | Verified in which stack? | GEMMAMONSTER / Arc 140V status |
| --- | --- | --- | --- |
| **Qwen3.6** | [Unsloth Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | Unsloth llama.cpp with `--spec-type draft-mtp`; Strata native engine supports this GGUF + MTP on NVIDIA/AMD | **Candidate**: GGUF MTP does **not** automatically become an OpenVINO draft IR. Verify export, token IDs, MTP tensors and GenAI support |
| **Qwen3.6** | [AEON-DFlash-Qwen3.6-35B-A3B](https://huggingface.co/AEON-7/AEON-DFlash-Qwen3.6-35B-A3B) | Published for Qwen3.6 and derivative checkpoints, using vLLM DFlash on GB10 | **Candidate**: model/arch/IR export and quality+memory proof required |
| **AEON Ornith** | **AEON DFlash** as above | Drafter author reports native support and Ornith results on DGX Spark | **Strong candidate**, but no measured Arc/OVMS acceptance data |
| **AEON Ornith** | **Base Qwen3.6 MTP head** | No reliable correctness guarantee across fine-tune and quantizations; separate MLX graft attempt reports a nonworking combination | **Unvalidated**: do not publish performance claims or enable by default |
| **Qwen3.8-Flash-Next** | Native checkpoint MTP head; Strata MTP | Strata reports 55–85% draft acceptance depending on prompt and hardware; engine-specific; draft uses quantized head | Research exemplar for draft/verify cache pressure |
| **Gemma4 26B-A4B** | Existing C6 speculative support where checkpoint-matched and already accepted | C6 known-good protocol is priority | Do not trade stable tool calling / slots for speculative speed |

Sources: [Strata Qwen3.6 models](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md), [Qwen3.6 concurrency and MTP](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md), [Unsloth MTP model](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), [AEON DFlash model](https://huggingface.co/AEON-7/AEON-DFlash-Qwen3.6-35B-A3B), [Strata Qwen3.8 quantization map](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md).

**Important discrepancies:** An older [Ornith serve script](https://github.com/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored/blob/main/serve_ornith.sh) says no MTP/DFlash drafter *at the time it was authored*. The later AEON DFlash checkpoint **does exist**. Do not use that stale script as proof that current DFlash is unavailable. Likewise, a failed [third-party Ornith MTP graft](https://huggingface.co/programmer-666/Ornith-1.0-35B-MLX-oQ7/blob/main/README.md) does not disprove compatibility of *all* MTP heads; it just means compatibility needs hard, checkpoint-specific acceptance tests.

## Serving-stack reality and multi-slot gates

- **C6 current detection**: `src/llm/servable_initializer.cpp` determines `MTP` when `openvino_mtp_model.xml` exists. `dflash_mode` and `eagle3_mode` `rt_info` markers identify DFlash/EAGLE3; arbitrary GGUF and vLLM drafter folders are not directly accepted.
- **Strata Qwen3.6**: [docs/QWEN36.md](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md) says solo MTP is distinct from multi-request batch MTP; `--batch-mtp` is an opt-in. Dedicated KV/recurrent-state resources per slot compete with expert residency. Thus **C1 acceptance is not proof of C4 acceptance**.
- **llama.cpp Unsloth route**: The [model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) documents `-np 1` and notes that `-np > 1` and `--mmproj` are not supported with MTP on that stated recipe. This must **not** be generalized to C6/GenAI or Strata: different engines have different concurrency paths.
- **AEON DFlash on GB10**: The [drafter card](https://huggingface.co/AEON-7/AEON-DFlash-Qwen3.6-35B-A3B) reports measured acceptance and a unified-memory ceiling: `max-num-seqs <=16` on Spark. That bound is **hardware/stack-specific**, not an Arc 140V recommended value.

## Acceptance requirements

- **Each target model**: checkpoints and tokenizer SHAs, model-specific tool-call tests (Gemma canonical protocol vs Qwen/Ornith agentic behavior), memory ledger, no cache corruption, actual expert routing preserved.
- **Drafter mode**: accepted tokens per verify, accepted ratio, time spent drafting vs verifying, extra KV/resident/draft buffers and **physical NVMe read bytes per committed output token**.
- **Multi-slot**: C1/C2/C4, plus C8 only when memory permits; control KV U4, prefix reuse, prompt mix, speculative width; compare wall time, fairness, p95 per-request TPOT and aggregate throughput.
- **No false positive**: If drafter cannot be exported to OpenVINO or scheduling fails with >1 slot, mark the stack explicitly unsupported; do not silently run plain and label it “MTP”.
- **Correctness**: lossless speculative decoding with target-side verification should preserve target-model sampling distribution, provided draft/verify implementation is correct; test token/logit parity and structured tool-call behavior.

## Minimum routing tests

```text
Gemma4 C6:       plain known-good, C1/C2/C4, tool protocol exact
Qwen3.6 base:    plain, native MTP, external DFlash; C1/C2/C4
AEON Ornith:     plain, AEON DFlash; base MTP only if compatible
Qwen3.8:         plain, native MTP with per-layer precision inventory
Qwen3-30B:       diagnostic OTD only
```

The validation target is not maximal token/s on one prompt; it is **accepted-token throughput and quality at the user's real multi-agent concurrency**.
