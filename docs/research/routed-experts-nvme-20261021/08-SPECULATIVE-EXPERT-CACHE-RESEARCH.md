# Speculative decoding, expert locality and NVMe working-set design

**Status:** research/benchmark contract, not an implementation patch.  
**Targets:** Qwen3.6, AEON Ornith, Qwen3.8 MTP, Gemma4 multi-slot C6 control.

## 1. Why speculation changes the expert cache

Speculative decoding separates a **drafter** from a **verifier/target**. A verification window proposes several positions, each possibly activating a different set of routed experts. A rejected proposal still costs some amount of **draft and/or verification computation and expert traffic**. Therefore caching only by *generated final tokens* loses the causal information required to predict misses and select eviction victims.

A drafter might have its own routed expert set (if a MoE draft model) or only dense layers / an MTP head; **do not assume the same expert IDs for draft and verify**. A native MTP head and an external DFlash have different compute footprints and memory layouts.

### Proposed accounting identity

```text
physical_nvme_bytes_per_committed_token
  = actual_NVMe_read_bytes_during_run / accepted_output_tokens

expert_work_per_committed_token
  = (verifier_expert_accesses + drafter_expert_accesses) / accepted_output_tokens

wall_time_per_committed_token
  = (draft_time + verification_time + cache_stalls + scheduling_overhead) / accepted_output_tokens
```

Record prompt prefill traffic separately from decode. Count *physical* NVMe reads, not only logical cache misses. Distinguish accepted output tokens from draft proposals and rejected candidates. Acceptance here must not be conflated with cache-hit ratio.

## 2. What each trace entry needs

```text
run_id, model_sha, draft_sha, quant_profile_sha, genai_sha, openvino_sha
slot_id, batch_id, sequence_id_hash, phase(prefill/draft/verify/commit)
round_id, proposal_position, accepted_flag, verified_length, committed_length
layer_id, expert_id, topk_weight, kernel_format, precision_profile_id
expert_slot_id, residency_state, cache_hit, load_reason(demand/prefetch)
physical_read_start, physical_read_length, read_us, transform_us, copy_us
kv_u4_bytes, prefix_hit_tokens, draft_state_bytes, target_state_bytes
time_ns, mode(plain/native_mtp/dflash), candidate_window, concurrency
```

Do **not** log private user prompts, filenames, local absolute paths or raw token text. Use deterministic synthetic fixtures and hashed request IDs.

A trace should preserve each exact router top-k decision in the order executed. Cache simulation must replay verifier draws and any separate drafter draws, **including rejected positions**, with real batch interleaving.

## 3. Cache policy distinctions by phase

- **Prefill** activates many experts across a large batch; maximize read coalescing and per-expert batching. Measure whether larger prompt chunks reduce per-token byte traffic or evict long-lived hot experts.
- **Plain decode** accesses one position per sequence per step; a hot expert may stay resident across many rounds.
- **MTP verify** processes several speculative positions; per-window union of experts can grow, but one loading round may serve multiple positions. Both positive locality (shared experts) and negative locality (rejected work) are possible.
- **External DFlash** adds a separate draft stage and memory ledger. Its own weights, KV and parallel kernels must not be treated as free.
- **Concurrent C2/C4** expands the union of expert IDs and increases KV U4 + per-slot draft state; could lower hit rate despite better disk queue utilization.
- **Prefix caching** reduces repeated prompt prefill and changes which expert keys are likely still hot when decode starts. Keep prefix hits as separate explanatory variable.

## 4. Required experiment matrix

| Axis | Values |
| --- | --- |
| Target checkpoint | Gemma4 26B-A4B, Qwen3.6-35B-A3B, AEON Ornith, Qwen3.8-Flash-Next |
| Draft mode | None; native MTP; external DFlash where supported, compatibility checked |
| Verify width | 1/control and small windows (2, 4, 6); larger only after measured benefit |
| Concurrency | 1, 2, 4; 8 if actual memory budget allows |
| Cache policy | Existing LRU, CLOCK, 2Q, routing-aware replay |
| Expert cache capacity | 10%, 25%, 50%, 75% of physically feasible bounded slots, not blind VRAM percentages |
| Prompt / answer | Short QA, code generation, repeated agent tool loops, long repository instructions, multilingual prose |
| Context | 1K, 8K, 32K, 64K; larger by strict memory admission |
| KV precision | Arc 140V **U4 fixed**; U8 only an explicit A/B experiment |
| Mode | Warm/cold cache, fixed and churned prefix, unbatched/batched MTP |

Report acceptance length, effective tokens/round, total tokens/s, p95/p99 TTFT and TPOT, reads per *committed* token, router expert union size, cache miss rate, prefetch precision, thermal stability and quality.

## 5. Strata observations to compare carefully

- [Strata Qwen3.6](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md): multi-conversation serving shares expert loads where possible; batched MTP is opt-in and adds a per-slot draft. It documents that more slot KV reduces expert-cache capacity. This maps directly onto our OVMS-slot concern.
- [Strata Qwen3.8 quant formats](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md): different *experts and layers* use different quantization formats. An average bytes-per-expert estimate is wrong: each `(layer, expert, projection)` has its own physical bytes and load/transform cost.
- [Strata Intel analysis](https://github.com/Niko1221/Strata/blob/main/docs/INTEL.md) shows native MTP acceptance can vary materially across code and prose. Intel B70 is **not** a 258V/Arc140V benchmark.
- [Unsloth Qwen3.6 MTP](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) is available, but its GGUF status does not prove OpenVINO exportability. [AEON DFlash](https://huggingface.co/AEON-7/AEON-DFlash-Qwen3.6-35B-A3B) provides a separate external model.

## 6. Placement/policy principles

1. Keep **target verifier demand misses highest priority**, ahead of speculative prefetch and low-utility draft reads.
2. Track **model generation and quant format per expert**: no silent reuse of a slot filled with a different checkpoint or packing.
3. Avoid duplicate loads of the same target expert requested by multiple verify positions or sequences, with a single-flight loading state and bounded slot leases.
4. Consider **window union size** in cache admission: a batch of six proposals may need more simultaneously-live experts than six isolated iterations.
5. Prematurely evicting target experts to prefetch low-confidence draft candidates can reduce accepted-token throughput. Use expected acceptance, demand probability, read cost and slot footprint in policy evaluation.
6. Reject windows safely on memory pressure; shrink speculative width, or fall back to plain decoding instead of swapping the OS.
7. For CPU expert lanes on P720, candidate kernels and cache spaces may be NUMA-local; their expert bytes/cost should be accounted separately from GPU paths.

## 7. Proposed acceptance gates

- **K0** model/drafter artifact compatibility, and real OTD kernel eligibility proven separately.
- **K1** plain model correctness and Gemma4 protocol known-good (C1/C2/C4).
- **K2** native MTP/DFlash output parity on each selected Qwen3.6/Ornith checkpoint; no silent fallback.
- **K3** instrumented routing trace with accepted/rejected speculative work; replay deterministic.
- **K4** policy improves *end-to-end accepted-token rate* and/or memory capacity without breaking p95 fairness.
- **K5** stable soak and fault injection with KV U4, prefix cache, multiple OVMS slots.
- **K6** candidate implementation only, mainline C6 runtime and accepted XGrammar lock untouched.

**Architecture hypothesis, not established fact:** the best residency policy for agentic Qwen3.6 might be *speculative-window-aware*, not only router-history-aware. This must be demonstrated by replay before adopting it.
