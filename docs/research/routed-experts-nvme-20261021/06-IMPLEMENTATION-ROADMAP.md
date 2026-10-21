# Implementation Roadmap: Routed Experts / MoE OTD for GEMMAMONSTER

**Status:** proposal / future implementation, **NOT an approved runtime patch**  
**Date:** 2026-10-21  
**Primary source:** [manual original research export](../../../routed-experts-research-original.md) (1,266 lines; commit `118148ce`).  
**Additional evidence:** `02-Xe2-CPU-DeepSeek-addendum-20261021.md`, `03-Strata-source-audit-20261021.md`, upstream OpenVINO v2026.4.0 sources.

## 0. Architectural decision

**Build on native OpenVINO GPU Offload-to-Disk (OTD). Do not implement an independent pager unless measurements and APIs require one.**

Verified in **OpenVINO 2026.4.0**, not only moving `master`:

- [`IExpertWeightProvider`](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/expert_weight_provider.hpp) already abstracts resident vs OTD experts.
- [`OffloadExpertWeightProvider`](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/expert_weight_providers.cpp) does expert-slot assignment, deduplication, LRU, and on-demand loading.
- [`lru_cache.cpp`](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/lru_cache.cpp) is the existing policy implementation.
- [`moe_otd_runtime.cpp`](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_runtime.cpp) performs positional file reads, per-tensor scale/ZP transpose and GPU buffer copies.
- [`moe_otd_runtime.hpp`](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_runtime.hpp) already declares `MOE_OTD_PERF_LOG` counters. **They print on normal process exit; they are not a per-request metrics API.**
- [`moe.cpp`](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/plugin/ops/moe.cpp) accepts offload for a supported compressed MoE path. **Do not assume every Gemma/Qwen/GLM model uses that kernel.**
- [`OFFLOAD_RATIO`](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/inference/include/openvino/runtime/intel_gpu/properties.hpp) is a real GPU property. In this source path, 1–99 enables OTD and **both 0 and 100 disable it**. 100 is **not** a valid "all disk" experiment.
- [C6 LM_CB initializer](https://github.com/DassaultFalconKing/gemmamonster_model_server_OVMS/blob/c6-head-mainline-20261020/src/llm/language_model/continuous_batching/servable_initializer.cpp) parses node `plugin_config` and passes properties to `ContinuousBatchingPipeline`. Validate the exact JSON spelling on the pinned binary before prescribing run flags.

**Responsibility boundaries:** OVMS owns API, request lifecycle and GenAI configuration; OpenVINO GenAI owns batching / KV; **OpenVINO Intel GPU plugin** owns actual MoE expert residency, weights, device buffers, OTD and kernel dispatch. A router trace from OVMS HTTP handlers would be insufficient.

## 1. Non-negotiable invariants

- Leave canonical **C6**, accepted XGrammar lock, staged runtime, Gemma4/Qwen parsers, tool generation policies and existing model deployments unchanged.
- Arc 140V **KV U4 is the user-verified known-good default**. KV U8 is an optional controlled comparison, never default. Prefix caching and scheduler are independent experiment axes.
- Explicitly isolate **KV quantization**, **expert weight storage format**, **expert compute precision**, **accumulation precision**. A Q4 pack does not imply INT4 XMX instructions are dispatched.
- 258V is 4P + 4 low-power E cores / 8 total threads, Xe2 XMX GPU, 32 GiB UMA: **no physical GPU-VRAM/host-RAM/SSD triple tier**.
- Resident dense/shared weights, KV U4, activations, OS+runtime and I/O staging must fit physical LPDDR before an expert arena is allocated. Reject admission instead of swapping.
- Expert routing top-k/weights, model-generation identity and decoding semantics must not be changed by a cache/prefetch optimization. Approximate CPU math requires an explicit separate mode and quality gate.
- No absolute private machine paths, credentials, internal addresses or per-user prompt bodies in public docs, commits or traces.

## 2. Workstreams, in order of dependency

| Stage | Scope | Deliverable | Gate / go-no-go |
|---|---|---|---|
| **P0. Pin + capability audit** | Exact C6 binary/OVMS/OpenVINO/GenAI/GPU driver/model/quantizer SHA; inspect `MOECompressed` eligibility and `OFFLOAD_RATIO` exposure on actual builds | Reproducibility manifest; support matrix for Gemma4, Qwen3-30B-A3B and later Qwen3.6/3.8 | OTD is confirmed by runtime counters; if unsupported, document the precise missing graph/kernel |
| **P1. Native OTD benchmark** | Qwen3-30B-A3B INT4; resident (if fits) vs ratio 20/40/60/80; C1/C2/C4, prompt 1K/8K/32K, 128/512 output, hot/cold; KV U4 fixed | Baseline JSON/CSV + graphs + exact log/metric provenance | Stable inference, no OOM, no swap storm, no protocol failures |
| **P2. Observability only** | Existing OTD perf counters, host I/O, kernel/profiler timings, GPU use, memory, request TTFT/TPOT; separate warm/cold and read vs transpose vs copy | Per-run OTD dashboard and timeline | Reconcile physical bytes, cache hit rate, storage waits, total end-to-end latency |
| **P3. Real router traces + offline policy simulator** | Record exact (layer, routed expert, token position, request id / batch boundary), no prompt content; replay actual LRU vs CLOCK/2Q/weighted-CLOCK across bounded arena sizes | Trace format, replay tool, deterministic reference fixtures | Candidate must reduce **miss bytes/critical-path stalls**, not only increase raw hit % |
| **P4. First low-risk optimization** | Evaluate pre-packed weight/scales/ZP layout vs current runtime transpose + nine-input loading; bounded buffers; optional path, parity reference | Data-layout prototype and A/B artifact | No wrong bytes, reduced wall cost in measured bottleneck, clean fallback |
| **P5. Policy and I/O experiments** | Candidate cache policy after P3, demand-first read scheduling, single-flight misses, bounded prefetch/staging, Windows overlapped I/O / Linux io_uring or positional fallback | Experimental GPU-plugin implementation with feature flags | P95/P99 latency not worse, read amplification bounded, no starved demand reads |
| **P6. CPU Experts separately** | Model-matched SIMD kernels on Ryzen AVX2, Xeon 6242 AVX-512 VNNI and Core Ultra 258V AVX2; NUMA + CPU/GPU overlap | Isolated kernel bench and quality report, no OVMS dependency | Admit CPU lane only if latency/energy or capacity actually improves |
| **P7. C6 candidate integration + reliability** | Use **separate, SHA-pinned OpenVINO candidate build** in an isolated runner; same OVMS API/harness; stage only after gates | Reproducible integration report / reversible feature flags | 30–60 min soak, integrity/fault tests, no Gemma/Qwen tool-call regression |
| **P8. Larger model feasibility** | 50–100B and later Qwen3.6/3.8/GLM variants only if architecture and memory budgets permit | Feasibility matrix, not a promise of TPS | Measured dense/KV footprint and SSD bytes/token within acceptable budget |

**P2 should precede any code that changes cache behavior.** P3 replay must precede P5 policy changes. P6 may run in parallel after P0.

## 3. Working hypotheses to falsify

1. **On-demand OTD costs more in repeated scale/ZP transpose + buffer copy than expected.** Existing OTD source explicitly does these transformations on misses. Measure proportion of TTFT/TPOT, do not assume disk-only bottleneck.
2. **Multiple reads per expert increase overhead.** Provider maps nine routed tensors (three weights, three scales, three zero-points). Verify coalescing opportunity against exact IR offsets.
3. **LRU suffers on locality / multi-request workloads.** Test from real router traces. The provider/cache structure is already in-place, making a policy prototype lower risk.
4. **Prefetch improves miss latency but may increase physical NVMe bytes/token.** Compare demand-only and layer-ahead hints; routed top-k must stay exact.
5. **258V may prefer INT8 XMX kernel to nominally lower-bit INT4 depending on batch geometry and dequant.** Test actual GPU dispatch, not theoretical TOPS.
6. **CPU lane on 258V can lose to GPU-only due to shared LPDDR/power contention** despite being parallel; more promising on dual Xeon / high-RAM hosts. Benchmark, never assume.
7. **Context growth competes with expert arena.** With KV U4 and prefix reuse fixed, measure expert-hit degradation at 32K/64K/128K/262K where valid.

## 4. Hardware-specific execution matrix

| Host | GPU/main path | CPU SIMD | Specific tests |
| --- | --- | --- | --- |
| **Core Ultra 7 258V / Arc 140V / 32 GiB UMA (Windows)** | OpenVINO GPU; native OTD; KV U4 | AVX2/FMA; 4P+4E topology | Expert hotset budget, dynamic LPDDR pressure, XMX INT4/INT8/BF16/FP16, CPU+GPU bandwidth contention, Windows asynchronous reads |
| **Older Ryzen (Zen1/Zen2/Zen3/Zen3+)** | OVMS CPU and optional available GPU | AVX2/FMA; SKU-dependent cache and bandwidth | Low-bit packed experts with vectorized GEMV, int8 activation accumulation vs fp32, thread scaling |
| **P720, 2× Xeon Gold 6242, 384 GiB RAM** | OVMS/OpenVINO CPU; optional P5000 only as distinct path | AVX-512 VNNI + AVX2/FMA; **no AMX, no AVX512_BF16** | Full-RAM baseline, NUMA-first expert placement, cross-socket migration costs, per-socket thread pools |
| **Xeon Scalable 3rd gen** | OpenVINO CPU | AVX-512 VNNI; some SKUs have BF16 instruction support | SKU-specific CPUID, BF16 vs INT8 GEMM |
| **Xeon Scalable 4th+** | OpenVINO CPU | AMX INT8/BF16 where OS exposes tile state | Batched GEMM / tile utilization; avoid invoking unsupported AMX |

**Do not directly import `glm-flash-lite` EXL3 Trellis kernels for DeepSeek MXFP4 or Qwen INT4.** Only portable *math skeleton / scheduling* is shared. Separate packer/dequant/epilogue implementations require real-weight numerical parity, including router weighting and model-specific SwiGLU clamp behavior.

Source for AVX2 operations: [`ft_n135.h`](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_n135.h), [`ft_core.h`](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_core.h). MXFP4 fidelity and arena: [DeepSeek V4 research](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/DEEPSEEK-V4.md).

## 5. Test design and mandatory evidence

**Dimensions** (freeze all but one per A/B):
- Primary checkpoint: Qwen3-30B-A3B INT4 OpenVINO; record exact architecture, weights and plugin version. Gemma4 26B-A4B only after kernel eligibility is checked; Qwen3.6/3.8 is follow-on.
- `OFFLOAD_RATIO`: 0 if resident fits, then 20/40/60/80; never 100; model must genuinely execute OTD.
- KV: **U4 fixed**; prefix cache enabled as a stable, measured configuration; evaluate scheduler/cache budgets in separate series.
- Workload: C1/C2/C4; prompt 1K/8K/32K, then larger context when safe; output 128/512; warm/cold; long agent tool-loop fixtures.
- At least 3 measured repetitions after warmup for screening; for borderline results repeat enough samples for confidence intervals. Pair prompts/seeds/context.
- Soak: 30–60 minutes; include warm-cache saturation and adversarial expert working sets.

**Minimal metric schema**:
```text
run_id, runtime_sha, openvino_sha, gpu_driver, model_sha, quant
offload_ratio, context_length, concurrent_requests, prompt_tokens, output_tokens
ttft_ms, tpot_mean_ms, tpot_p50_ms, tpot_p95_ms, tpot_p99_ms
prefill_tps, decode_tps, total_tps, max_ram_bytes, swap_bytes
expert_unique_per_token, expert_cache_hits, expert_cache_misses, evictions
disk_read_bytes, disk_reads, disk_read_p50_us, disk_read_p95_us
transpose_us, gpu_copy_us, gpu_stall_us, cpu_busy, gpu_busy
prefix_cache_hit_tokens, io_queue_depth, prefetch_precision
logits_error_max, kl_divergence, top1_agreement, tool_loop_status
```
Tag unavailable fields as `NOT_EXPOSED`; never fake per-request measurements by scaling global exit-time counters.

**Quality gates**:
- Exact top-k/router decisions and tensor checksums, no stale-generation expert.
- Numerical equivalence to same-kernel resident path within documented tolerance; approximate CPU execution has separate threshold and opt-in.
- Named/required/auto/parallel tool calls and multi-turn reasoning transitions retain C6 acceptance behavior.
- No OOM, swap storm, missing experts, deadlocks, double-loading, or use-after-eviction.
- New path passes independent benchmark; production admission requires meaningful end-to-end benefit, not just improvement in a synthetic microbenchmark.

## 6. Engineering seams and safe staging

**Existing C6 interface (observe/configure only):**
- `src/llm/language_model/continuous_batching/servable_initializer.cpp`: node plugin config and GenAI scheduler settings.
- `src/llm/language_model/continuous_batching/servable.cpp`: request-scoped perf metrics.
- `src/llm/language_model/continuous_batching/llm_executor.hpp`: pipeline/cache runtime metrics.
- Keep parser/tool-generation modules, C6 xgrammar lock and known-good binaries outside the experiment.

**OpenVINO GPU plugin candidate seam:**
- `expert_weight_provider.hpp` / `expert_weight_providers.cpp`: optional residency/admission extensions.
- `lru_cache.{hpp,cpp}`: alternative replacement strategies behind flags, preserving LRU default.
- `moe_otd_runtime.{hpp,cpp}`: measurement, loading, staging and packed-layout options.
- `moe.cpp`: capability/generation checks; **avoid changing routing math**.
- Related OTD unit tests: `src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_*.cpp`. Extend these before end-to-end serving.

**Suggested independent branches/worktrees (names are proposals, not created):**
```text
docs/moe-nvme-experts-research-20261021         existing documentation
research/ov-gpu-otd-observability-2026          external OpenVINO pinned candidate
research/otd-router-trace-and-cache-sim-2026    independent replay tools
research/cpu-expert-simd-2026                   AVX2/VNNI proof of concept
integration/c6-otd-candidate-2026               optional last-stage isolated candidate
```

**First operational request for a local agent:** produce read-only P0/P1 setup, source/build manifest and run matrix. It must **not** modify C6 or dependencies, must not push local absolute paths, and must not create a new pager before confirming the native OTD behavior.

## 7. Decision record: do not build until the evidence supports it

- If native OTD with safe KV U4 + prefix caching meets capacity and latency goals, **stop at native path** and deliver a deployable profile.
- If misses dominate, prioritize cache policy + prefetch; if conversion/copy dominates, prioritize packed expert layout; if storage contention dominates, investigate async I/O.
- If kernel execution dominates, optimize Xe2 compute precision/shape, not disk tier.
- If CPU overlap is positive on the P720 but not the 258V, expose CPU lane **only** on suitable hardware, not as universal dispatch.
- If an upstream extension is possible, prefer minimal upstreamable changes over a permanently divergent OpenVINO fork.

**Hard stop:** any attempt to silently replace accepted C6, staged runtime, model checksum, tool-call parser semantics, XGrammar lock, or the KV U4 known-good configuration.

## 8. Milestone order

```text
P0 environment + model eligibility
   |
P1 native OTD reproducible A/B ---- P2 profiling and counters
   |                                      |
   +--------------- P3 router traces ----+
                       |
                 offline simulator
                       |
           bottleneck-driven choice
              /         |          \
       cache policy  packed IO   GPU precision
              \         |          /
                    P5 prototype
                       |
                  P7 isolated C6 acceptance
                       |
                  P8 larger MoE

P6 CPU SIMD research starts independently after P0; joins only if quality and wall-clock gates pass.
```

**Primary reference is retained byte-for-byte in the manual export.** This roadmap is an additional synthesis based on code inspection; it does not supersede the original report.
