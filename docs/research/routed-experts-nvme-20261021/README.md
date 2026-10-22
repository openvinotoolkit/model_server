# Routed Experts, NVMe offload, Strata, Xe2 и CPU experts

**Research packet — 21 October 2026 · documentation only**

Target: GEMMAMONSTER / OpenVINO Model Server. Platforms: Intel Arc 140V + Core Ultra 7 258V (32 GiB unified memory), AVX2 Ryzen, Xeon Scalable 2nd generation and newer, including dual Xeon Gold 6242.

## Contents and provenance

| File | Contents | Provenance | Status |
| --- | --- | --- | --- |
| [01-ARC-140V-research-20261006.md](01-ARC-140V-research-20261006.md) | Earlier investigation: OVMS, Arc 140V, MoE disk offload, runtime choices | Existing research document of 2026-10-06; publication copy changes one example machine-specific path to `<MODEL_REPOSITORY>` | Historical, unchanged technical conclusions |
| [02-Xe2-CPU-DeepSeek-addendum-20261021.md](02-Xe2-CPU-DeepSeek-addendum-20261021.md) | KV U4, XMX numerical formats, DeepSeek disk loading, AVX2/AVX-512/AMX, expert kernels | Original previously generated Markdown addendum | Add-only |
| [03-Strata-source-audit-20261021.md](03-Strata-source-audit-20261021.md) | Intel SYCL support limitations and portable architectural mechanisms | Newly assembled source-based companion | Add-only |
| [04-OVMS-integration-research-notes.md](04-OVMS-integration-research-notes.md) | Evidence-oriented proposal, implementation seams, experiment matrix and gates | New synthesis of known findings, **not** claimed to be original Deep Research text | Add-only |
| [05-SOURCES-AND-GAPS.md](05-SOURCES-AND-GAPS.md) | Sources, known limitations, original-report availability and missing evidence | Package provenance/manifest | Add-only |

### Important: missing full original Deep Research export

The primary long-form Deep Research report, **«Интеграция Routed Experts и механизмов GLM-Flash-Lite с NVMe в ОВМС на Intel Arc 140V»** (21 October 2026), was produced in the research app, but its *complete, exact text was not available as an exportable file* in the present GitHub publishing context. Consequently, **this package does not claim to contain an unmodified copy**. Its known conclusions are represented only as newly authored research notes where explicitly marked. The original is **not deleted, shortened, replaced, or silently reconstructed**. See [05-SOURCES-AND-GAPS.md](05-SOURCES-AND-GAPS.md).

### Hardware and correctness baselines

- **Arc 140V KV U4** is user-established, repeatedly validated and preferable to U8. Preserve U4 as baseline. Prefix cache and scheduler are separate axes.
- Xe2 XMX supports low-bit integer matrix modes and BF16/FP16, but the *actual winner* requires per-shape, per-kernel benchmarks; do not infer performance solely from peak operations.
- Treat CPU expert execution as a candidate parallel lane, not an established win on unified-memory Lunar Lake.
- The dual Xeon Gold 6242 machine has AVX-512 VNNI but neither native AMX nor AVX512_BF16; it needs its own NUMA-aware tests.
- All public sources should be interpreted at **their actual model version, hardware and operating system**.

### Isolation

This documentation branch contains no runtime, parser, scheduler, tokenizer, XGrammar or dependency-lock modifications. The mainline C6, staged runtime and accepted dependency lock remain untouched.

**Publication rules:** no local machine directory map, local secrets, credentials, private IP addresses, or local absolute checkout paths. Use placeholders in commands.

## Update: exact original export is now available

On 2026-10-21 the complete original Deep Research export was added manually as [`routed-experts-research-original.md`](../../../routed-experts-research-original.md), commit `118148ce450cb9805eb2a193113b6fa77a87b563`. The earlier "missing original" statements above record the publishing state *before* that manual export; **the gap is now resolved**. The original is kept at its manually committed location, unmodified.

**Implementation planning:** [`06-IMPLEMENTATION-ROADMAP.md`](06-IMPLEMENTATION-ROADMAP.md). This supersedes earlier assumptions that a new Expert Residency Manager/LRU must be implemented from scratch: OpenVINO 2026.4.0 already has those components. Runtime C6 stays pinned.

## Model/drafter priority correction (2026-10-21)

**Golden baseline is not old Qwen3-30B-A3B:** use **Gemma 4 26B-A4B** for accepted GEMMAMONSTER protocol, slots and tool calling, **Qwen3.6-35B-A3B plus AEON Ornith** for real agentic model and NVMe OTD, and **Qwen3.8-Flash-Next** for MTP and per-layer mixed-quant memory pressure. The older Qwen3-30B-A3B is a diagnostic reference only.

Available draft candidates are now identified, not hypothetical: [Unsloth Qwen3.6 MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), [AEON DFlash for Qwen3.6 and Ornith](https://huggingface.co/AEON-7/AEON-DFlash-Qwen3.6-35B-A3B), native Qwen3.8 MTP. C6 recognizes OpenVINO MTP and DFlash formats but cannot directly load an arbitrary GGUF drafter without compatible export. Fine-tune-specific draft acceptance must be measured.

- [07 — Model and drafter baselines](07-MODEL-AND-DRAFTER-BASELINES.md)
- [08 — Speculative Expert Cache and NVMe study](08-SPECULATIVE-EXPERT-CACHE-RESEARCH.md)
- [06 — Revised implementation roadmap](06-IMPLEMENTATION-ROADMAP.md)

## Architecture plan v2 (2026-10-22, docs-only)

**New architecture entry point:** [09 — GEMMAMONSTER architecture plan v2](09-ARCHITECTURE-PLAN-V2.md). Includes source-grounded OVMS/GenAI/GPU ownership, hot/warm/cold per-layer expert layout, UMA ledger, plain vs checkpoint-gated MTP/DFlash, target-demand-first OTD, native LRU vs proposed cache/CPU experiments, v1-to-v2 migration and coding-agent P0–P8 acceptance gates.

- [Root architecture flowchart v2](../../../architecture_flowchart_v2.mermaid) — serving, draft/verify, OTD and conditional extensions.
- [10 — Cache and physical-memory layout v2](10-CACHE-MEMORY-LAYOUT-V2.mermaid) — pinned non-expert, hot per-layer routed experts, warm host/page cache (same LPDDR), cold NVMe, KV U4 and slot admission.
- [Original root flowchart v1](../../../architecture_flowchart_graph.mermaid) remains unchanged for provenance.

**Evidence boundary:** v2 synthesizes the research already published in this branch through commit `d2954601`. The newly requested Deep Research hot/cold architecture synthesis was started but its final report was not available as an export at publication time; do not attribute v2 text to that unreceived report. Future research arrivals should be added as dated supplements, not overwrite the manual original or stable C6.
