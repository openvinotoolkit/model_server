# Routed Experts NVMe / CPU-lane integration: research notes

**Дата:** 2026-10-21  
**Характер:** самостоятельный конспект подтверждённых источниками направлений и proposed acceptance gates. **НЕ исходный полный Deep Research отчёт.**

## 1. Объект и инварианты

GEMMAMONSTER на базе OVMS, Arc 140V/Core Ultra 7 258V (32 GB UMA), AVX2 Ryzen, Xeon Scalable 2nd generation and newer; сравнение штатного OpenVINO disk offload против candidate routed expert pager. Цель: MoE с рабочим набором экспертов меньше полного checkpoint, NVMe immutable source, опциональный CPU inference.

Инварианты:
- no changes to main C6, accepted XGrammar dependency lock, staged runtime;
- measured-good Arc 140V KV U4 remains default, not U8;
- dense weights / activations / KV occupy real system memory; expert streaming does **not** make these vanish;
- keep exact model semantics, routing top-k, logits and tool calling unless an explicit approximate mode is quality-gated.

## 2. Конкурирующие проекты: в чём реально различаются

| Подход | Формат хранения / backend | Экспертная резидентность | Переносимая идея | Не переносить напрямую |
| --- | --- | --- | --- | --- |
| OVMS / OpenVINO native | OpenVINO graph / plugin | Built-in disk MoE offload, версия и конфигурация важны | **Baseline**, measure before new code | Не считать наличие фичи автоматически подтверждённой для каждого Qwen/Gemma |
| GLM-Flash-Lite | CUDA/exllamav3, EXL3 Trellis | VRAM CLOCK + RAM exclusive tier + NVMe store; CPU AVX2 lane | Arena, async reader, CPU/GPU overlapped compute, prefetch, metrics | CUDA kernels, PCIe assumptions, Trellis decode для MXFP4 |
| DeepSeek V4-Flash MLX | Apple MLX; MXFP4 routed, higher-precision dense | Device LRU + parallel `pread` | Read-only packed experts, LRU, overlap, integrity | Apple MLX-specific kernels and published t/s on 48 GB |
| Strata | Custom CUDA/HIP; experimental Intel SYCL | GPU resident/pinned mirror/streaming; Intel discrete-card work | Residency planning, parity tests, staging | Current SYCL port to Windows Arc 140V and UMA assumptions |
| experts4bit-qlora | PyTorch/CUDA, NF4/MXFP4 | Native MXFP4 arena with lazy residency | MXFP4 reader, dequant/reference, quality tests | Reinterpret Qwen or GLM bytes as DeepSeek MXFP4 |

Sources:
- [OpenVINO release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html), [PR 36202](https://github.com/openvinotoolkit/openvino/pull/36202)
- [GLM disk/CPU architecture](https://github.com/sybil-solutions/glm-flash-lite/blob/main/docs/how-it-works.md)
- [DeepSeek SSD cache](https://github.com/ssd-moe/deepseek-v4-flash-mlx/blob/main/oracle/offload_cache_v4.py), [pread](https://github.com/ssd-moe/deepseek-v4-flash-mlx/blob/main/oracle/pread_loader.py)
- [DeepSeek V4 mathematical fidelity / MXFP4 arena](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/DEEPSEEK-V4.md)
- [Strata Intel Arc documentation](https://github.com/Niko1221/Strata/blob/main/docs/INTEL_ARC.md)

## 3. Hardware execution lanes

| CPU | Supported high-value paths | Constraint |
| --- | --- | --- |
| Older Ryzen Zen/Zen2/Zen3/Zen3+ | AVX2 + FMA; quantized GEMV or small expert GEMM | Memory bandwidth per core, cache locality; do not assume AVX-512 |
| **Dual Xeon Gold 6242 (Cascade Lake)** | AVX2, AVX-512, AVX-512 VNNI, BF16 through conversion/FP32 | no AMX or AVX512_BF16; critical to pin NUMA-local slots and thread pools |
| 3rd gen Xeon Scalable | AVX-512 VNNI; **some SKUs** BF16 instructions | Feature detection per actual CPU |
| 4th+ Xeon Scalable | VNNI / AMX INT8 + BF16 when exposed by OS and silicon | OS tile state, AMX gating, batching matter |
| **Core Ultra 7 258V** | CPU AVX2 for cold experts; Xe2 XMX GPU | Shared LPDDR, shared power, CPU/GPU resource contention |

Key prototype seams in [GLM ft_core.h](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_core.h), [ft_n135.h](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_n135.h): AVX2 shuffle, integer `vpmaddwd`, per-block scales, FMA, clamped SwiGLU, persistent pool. This is **EXL3 Trellis math**. Porting to DeepSeek MXFP4 or Qwen GGUF I-quants requires a *new decoder and parity tests*; cannot reuse bytes/lookup tables indiscriminately.

## 4. Candidate memory manager for UMA

1. Keep dense/non-MoE permanently resident only when memory budget allows.
2. Measure actual KV U4 footprint, activations and prefix-cache use, then reserve safety headroom.
3. Immutable packed NVMe expert store with manifest, per-record hashes, offset alignment.
4. Asynchronous read queue into bounded shared-memory staging, no unlimited page cache duplication.
5. Separate CPU vs GPU lane planner; first measure *cost model*, then make admission decisions.
6. Hot expert cache, bounded eviction, monotonic landed flags; never advertise expert resident before complete DMA/IO.
7. Prefill and decode use separate strategies; speculative draft+verify may multiply expert working set.

Potential memory equation (accounting identity, not measured capacity):

```text
peak_shared_RAM =
    resident_dense + resident_expert_hotset + staging
  + KV_U4 + activations + runtime + OS + transient_workspace
```

The compact formula alone cannot predict memory without actual tensor accounting on each model.

## 5. Benchmark and acceptance matrix

**A. Baseline, no experimental pager**
- Verify existing OVMS GPU profile with KV U4, prefix cache, tool loops; exact version/driver/model SHA.
- Reproduce native offload on same hardware, same prompts, same token sampling policy.

**B. Xe2 numeric kernel sweep**
- Weight formats: INT4, INT8, FP8, BF16, FP16 according to *actually supported kernels*.
- Task geometry: decode `M=1` GEMV, short batched MoE `M=2/4/8`, prefill batch; shape-specific XMX occupancy.
- Inspect real kernel dispatch, dequantization overhead, CPU/GPU bandwidth, accumulation error.
- Report TTFT and end-to-end tok/s, not merely kernel theoretical TOPS.

**C. CPU experts**
- Run AVX2 reference on 258V and older Ryzen; AVX-512 VNNI on Xeon 6242; AMX only where CPUID+OS permit.
- Compare CPU-only cold lane, GPU-only, overlapped; NUMA pinning and cold expert distribution on P720.
- Log cache hits, bytes NVMe/RAM, p50/p95 read stalls, CPU/GPU duty, power/thermals, correct logits.

**D. Context and agentic**
- KV U4 remains fixed baseline while checking prefix caching and scheduler interactions.
- Long context 32K/64K/128K/262K only if model + memory budget permits; fail cleanly on insufficient capacity.
- Repeated multi-turn tool calls, named/required/auto/parallel, reasoning transitions; no regressions.

## 6. Stop conditions

- Any change to pinned C6, staged runtime, current dependency lock, existing accepted parser policy.
- Silent logits/top-k mismatch for a mode claimed exact.
- OOM, paging storm, unrecoverable executor reset, corrupted expert loads.
- End-to-end slowdown vs native OpenVINO baseline without an explicitly stated memory-capacity benefit.
- Claimed platform support without a runnable reproduced command and source/benchmark artifact.

**Design decision:** baseline first; proof of expert-cache pressure second; kernel experiments third; OVMS integration only after independent correctness and performance gates.
