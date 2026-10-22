# GEMMAMONSTER — архитектурный план v2: MoE hot/warm/cold, NVMe OTD и speculative-aware cache

**Дата:** 2026-10-22  
**Статус:** **research architecture / proposed design; NOT deployed, NOT an implementation patch**  
**Основа:** опубликованный research packet в `docs/research/routed-experts-nvme-20261021` до commit `d2954601ea0e889de5631a059f26223aa074cd41`, исходный вручную сохранённый [Deep Research export](../../../routed-experts-research-original.md), [roadmap 06](06-IMPLEMENTATION-ROADMAP.md), [model/drafter matrix 07](07-MODEL-AND-DRAFTER-BASELINES.md), [cache research 08](08-SPECULATIVE-EXPERT-CACHE-RESEARCH.md), [Strata audit 03](03-Strata-source-audit-20261021.md) и [OVMS notes 04](04-OVMS-integration-research-notes.md).  
**Ограничение provenance:** новый Deep Research о раскладке hot/cold слоёв и синтезе v2 был запущен, но его завершённый текст **не был доступен как экспорт в момент этого коммита**. Поэтому этот документ — собственный проверяемый синтез доступных файлов, **не выдаваемый за полученный финальный отчёт**. При поступлении отчёта добавлять датированный addendum, не переписывать оригинал.

## 1. Схемы и смысл обозначений

- [Основной flowchart v2](../../../architecture_flowchart_v2.mermaid) — OVMS / GenAI / Intel GPU plugin, plain vs MTP/DFlash, native OTD и proposed experiments.
- [Раскладка кэша и UMA](10-CACHE-MEMORY-LAYOUT-V2.mermaid) — pinned, hot, warm, cold, demand/prefetch, единый бюджет LPDDR.
- [Исходная схема v1](../../../architecture_flowchart_graph.mermaid) оставлена без изменений как исторический вход для migration review.

**Легенда:** сплошная стрелка и голубой блок означают текущий контракт serving или *подтверждённую исходниками способность OpenVINO*; это **не доказательство**, что конкретный checkpoint попадает в OTD kernel. Пунктир/жёлтый блок означает условную ветвь, будущую политику, экспортный gate или эксперимент. Зелёные блоки обозначают физическую память/хранилище, не дополнительный уровень VRAM.

### Architectural decision record (ADR-v2)

1. **Сохраняем C6 как golden baseline.** Не менять stable C6, XGrammar, parser, tool-call protocol, staged runtime, dependency locks, model/tokenizer/quant checkpoint. Публикация v2 — **только docs**.
2. **Первый baseline — нативный OpenVINO Intel GPU OTD.** В upstream OpenVINO **2026.4.0** уже существуют `IExpertWeightProvider`, `OffloadExpertWeightProvider`, LRU, чтение весов и преобразования. Не создавать независимый pager до трасс, воспроизводимых bottleneck и доказанного выигрыша.
3. **Три основные архитектурные линии:** **Gemma 4 26B-A4B** (точный C6 tool calling / multi-slot), **Qwen3.6-35B-A3B** (главный agentic OTD target), **Qwen3.8-Flash-Next** (MTP / mixed-quant reference). **AEON Ornith** — отдельный checkpoint/derivative Qwen3.6 с собственной маршрутизацией и draft acceptance, а не четвёртая базовая архитектура. Старый Qwen3-30B-A3B — только диагностический контроль.
4. **Всегда есть plain decode.** MTP/DFlash разрешены лишь после checkpoint/tokenizer/export/IR/GenAI/slot compatibility gate. GGUF drafter из другого runtime не становится автоматически OpenVINO IR.
5. **Target verifier demand > drafter demand > speculative prefetch.** В baseline есть native on-demand LRU; отдельные приоритетные очереди, bounded prefetch, cancellation и cross-slot single-flight/leases — **предложения**, требующие инструментирования и отдельной реализации. Не подменять demand miss «prefetch miss».
6. **Реальный бюджет Arc 140V — один пул 32 GiB LPDDR.** GPU-readable USM, host DRAM, OS page cache, staging, dense, KV, draft и hot experts конкурируют за **одни физические байты**. NVMe отдельно; «VRAM + RAM + SSD» как три независимые ёмкости на 258V запрещено.
7. **Exact routing and numerical parity.** Cache, prefetch и размещение не изменяют router top-k, logits и sampling distribution. CPU lane и альтернативные quant/kernels допускаются только после parity gate.

## 2. Ownership boundaries и неизменяемые контракты

| Слой | Текущее владение | Вход/выход и инвариант | Новая работа, только после gate |
| --- | --- | --- | --- |
| **OVMS / C6** | HTTP/OpenAI-compatible API, tool schemas, named/required/auto/parallel tool calling, XGrammar, request lifecycle, конфигурация модели | Не меняет точность router или commit semantics; multi-slot C1/C2/C4 | Admission feedback и per-request telemetry **без** изменения protocol |
| **OpenVINO GenAI** | Continuous batching, target decode/verify, KV/prefix, speculative strategy when supported | Отдельные drafter и verifier states; корректный commit/rollback; KV U4 baseline | Mode/slot memory accounting, draft-window control и acceptance traces |
| **Intel GPU plugin** | MoE graph lowering, eligible compressed expert kernels, expert provider, resident/OTD LRU, file reads, GPU buffers | `OFFLOAD_RATIO` только на реально подходящем graph path; корректные weights/scales/ZP | Trace hooks, alternative policy, bounded I/O, prefetch, format packing |
| **Storage / OS** | NVMe packed weights, page cache, filesystem, physical LPDDR | Immutable/checksummed checkpoint; никаких скрытых копий или paging storm | Manifest/offset checks, physical read-byte telemetry |
| **CPU expert lane** | **Не часть принятого OTD baseline** | SIMD ISA/NUMA/kernel parity и cost model отдельно | Эксперимент на 258V AVX2, Ryzen AVX2, P720 Xeon Gold 6242 AVX-512 VNNI; no AMX on 6242 |

**Существующий код:** [C6 GenAI initializer](https://github.com/DassaultFalconKing/gemmamonster_model_server_OVMS/blob/c6-head-mainline-20261020/src/llm/language_model/continuous_batching/servable_initializer.cpp) передаёт GPU properties в GenAI pipeline; [C6 draft detection](https://github.com/DassaultFalconKing/gemmamonster_model_server_OVMS/blob/c6-head-mainline-20261020/src/llm/servable_initializer.cpp) распознаёт `openvino_mtp_model.xml` и маркеры `dflash_mode`/`eagle3_mode`. **Detection ≠ успешный export или запуск.**

## 3. Раскладка слоёв, экспертов и точности

| Компонент | Placement policy | Статус / проверка |
| --- | --- | --- |
| Embeddings, attention, routers, norms, output head, dense/shared experts | **Pinned / resident при выполнении memory gate**, не смешивать с pageable routed experts | Уточнить физические размеры по фактическому IR/checkpoint; pinned не означает безусловно помещается |
| Routed experts `(checkpoint, layer, expert, projection, quant_profile)` | **Hot**: per-layer GPU-readable resident pool, текущая политика native OTD LRU; набор `H_l` определяется только реальными routing traces | Не придумывать номера «горячих слоёв»/экспертов; приоритет динамический для prefill/decode, tool loops, языка, слотов |
| Неактивные expert pages / OS filesystem cache | **Warm**: физические страницы того же LPDDR, возможно повторное использование при read; **не второй бесплатный RAM tier** | Не считать OS page cache и staging дважды; замерить actual resident set |
| Packed expert tensors, scales, zero points | **Cold**: immutable NVMe backing store с offsets, manifest, hash, format metadata | Проверять endianness, alignment, packing и checkpoint generation |
| Drafter / native MTP head / external DFlash | Отдельный ledger и route/weights при наличии; target verifier cache нельзя приравнивать к draft cache | Native MTP vs DFlash и checkpoint-specific acceptance; C1 не доказывает C4 |
| KV/prefix/activations | KV **U4** как Arc baseline; unique physical KV pages, per-slot target/draft state, prefix reuse, bounded workspaces | U8 только отдельный A/B; 8K/32K/64K и C1/C2/C4 admission |

**Формат ключа:** `(model_checkpoint_sha, layer_id, expert_id, projection={gate|up|down|other}, quant_profile_sha, packing_version)`. Реализация может загружать один «expert bundle» из нескольких tensor records, но accounting должен различать физические веса, scales, ZP, padding и временные буферы. **Нельзя** усреднить размер одного эксперта по всему checkpoint и затем считать все слои одинаковыми.

**Per-projection mixed quant:** для `gate/up` и `down` допускаются разные битности, включая чувствительные слои с повышенной точностью; router/shared/attention/head обычно требуют отдельного профиля. Форматы GGUF Q*, EXL3, MXFP4, INT4, INT8, FP8, BF16/FP16 **не взаимозаменяемы** и **не подтверждают** поддержку соответствующего Xe2 kernel. Реальная схема задаётся checkpoint manifest, exporter, OpenVINO IR и kernel dispatch. Для Qwen3.8 Strata публикует неоднородные форматы, но **не** подтверждает их прямой запуск в OVMS/Arc 140V.

**Accounting identities (не оценка реальных MiB):**

```text
B_UMA_physical >= B_OS_and_runtime + B_pinned_non_experts
                + B_hot_expert_unique + B_warm_page_cache_unique
                + B_staging_and_transforms + B_KV_U4_unique_active_and_prefix
                + B_drafter_and_its_KV + B_activations_and_workspace
                + B_safety_margin

B_expert_hot_max = max(0, B_UMA_physical - sum(all other unique physical allocations))

B_expert_record = B_weight_payload + B_scales + B_zero_points
                + B_padding + B_resident_metadata
B_expert_bundle = sum(B_expert_record for actual projections in that bundle)

physical_NVMe_bytes_per_committed_output_token =
    measured_physical_NVMe_read_bytes / actual_committed_output_tokens
```

При дефиците headroom сначала ограничить prefetch/staging, затем speculative window и/или активные слоты, перейти в plain при необходимости; отказать в admission, если даже plain не помещается. **Не** разрешать скрытый OS swap. Для P720 384 GiB NUMA учёт отдельный, locality по socket и CPU thread pools; он не превращается в UMA 258V.

## 4. Точная последовательность decode / verify / cache

1. **P0 compatibility:** определить checkpoint, tokenizer, quant manifest, OVMS/OpenVINO/GenAI/driver SHA, доступный GPU graph lowering. Только `MOECompressed`/поддерживаемый compressed path плюс корректный `ov::weights_path` могут доказать native OTD. В OpenVINO 2026.4.0 для указанного пути `OFFLOAD_RATIO` **1–99 включает**, `0` и `100` **отключают** OTD. Отдельно проверить JSON plugin config на закреплённой сборке.
2. **Admission:** C1/C2/C4, контекст, KV U4/prefix, model/drafter bytes, OS headroom, bounded hot arena/staging; выбрать plain или разрешённый MTP/DFlash. **Никакого** silent fallback с меткой «MTP».
3. **Plain:** target вычисляет один очередной шаг, router выдаёт exact top-k per layer. На hit используем resident expert; на miss target **demand** получает приоритет и должен завершиться до dependent compute.
4. **Speculative:** drafter строит proposal window со **своими** весами/KV/маршрутами (если MoE). Target verifier оценивает proposed positions с **собственным** router и expert working set. Dedup и admission учитывают union window, но не меняют router top-k.
5. **Commit:** target принимает корректный prefix; при rejection применяет **target-corrected token** согласно алгоритму, откатывает/синхронизирует draft и target KV/state; при полном acceptance допускается алгоритмический bonus token, если поддерживается. Учитываются **все реально committed** токены, отдельно rejected proposals и их затраты. Не утверждать «reject -> commit accepted proposals» как универсальную ветвь.
6. **Storage:** native provider LRU / positional reads / scale-ZP transpose / GPU copy уже присутствуют upstream. Candidate scheduler **позже** может разделить target demand, draft demand и bounded low-priority speculative prefetch, с cancellation, single-flight across slots, slot leases, integrity и bounded staging. Сначала сравнить с нативным dedup/LRU.
7. **Feedback:** замерить physical read bytes и latency per committed token, acceptance/window, actual route hotness, TTFT/TPOT p95/p99, C1/C2/C4 fairness, OS memory pressure, tool-call protocol parity.

## 5. Что уже существует, а что остаётся гипотезой

| Механизм | Source-supported upstream/current | Proposed / не утверждать как deployed |
| --- | --- | --- |
| `OffloadExpertWeightProvider`, LRU, on-demand disk reads | **Да**, OpenVINO 2026.4.0, для eligible MoE graphs | Model-by-model support Qwen3.6/Gemma4/Ornith/Qwen3.8 пока **не доказан** |
| `MOE_OTD_PERF_LOG` | **Да**, run/process-exit counters | Per-request/per-slot trace, OS physical NVMe accounting, router events |
| C6 MTP/DFlash strategy detection | **Да**, C6 code | Конкретный GGUF/AEON drafter export и correct C4 GenAI execution |
| GPU expert LRU | **Да**, native baseline | CLOCK/2Q/GDSF, routing-aware cache, per-layer quotas, checkpoint-keyed extended arena |
| CPU SIMD experts, INT4 Xe2 special kernels | Аппаратные ISA/форматы требуют проверки конкретного dispatch | **Эксперименты**, не считать уже реализованными |
| Single-flight/dedup | Native provider уже содержит свою дедупликацию и slot assignment | Cross-request priority/lease/cancel semantics и new IO scheduler требуют аудита/патча |
| Layer hotness | Концепция из Strata и research | **Нет достоверного фиксированного списка layer/expert IDs без трасс** |

## 6. Миграция из `architecture_flowchart_graph.mermaid` (v1)

| Старый узел / допущение | v2 |
| --- | --- |
| `Scheduler -> Draft MTP/DFlash` без альтернативы | `Admission -> compatibility gate -> plain OR checkpoint-specific draft` |
| `Candidate window -> Target verify` для всех запросов | Раздельные plain target step и speculative verifier window |
| `Expert miss -> Routing-aware prefetch queue` | **Target demand miss -> native OTD**; speculative prefetch только отдельным gated low-priority экспериментом |
| `Protected / Probation hit` как будто уже native | **Native LRU**; protected/probation только после replay и feature flag |
| `Xe2 INT4/INT8` без kernel proof | Actual exported quant + actual GPU kernel dispatch + numerical parity |
| `CPU overlap wins -> AVX2/AVX-512` как runtime policy | CPU lane отдельная экспериментальная ветвь с ISA/NUMA/energy/parity gates |
| `Draft rejected -> Commit accepted tokens` | Accepted prefix, target-corrected token, rollback/sync, optional full-accept bonus, actual commit |
| `KV U4 + prefix -> Draft` без бюджета | One-UMA ledger с OS, pinned, expert, KV, prefix, draft, staging, C1/C2/C4 |
| `Source: NVMe or staging/USM` как независимые ёмкости | NVMe separate; staging/USM/page cache/CPU DRAM — **одна LPDDR** |
| «Горячие слои» заданы вручную | `H_l` per layer определяется replay трасс, без выдуманных expert IDs |

## 7. Coding-agent handoff: пакеты и gates

**Правило:** read-only аудит сначала; код только в **новых isolated candidate branches/worktrees** после P0. **Никогда** не коммитить в `c6-head-mainline-20261020`, не менять `src/llm`/XGrammar/lock на docs branch. Весь evidence содержит SHA checkpoint/driver/OpenVINO, commands с безопасными placeholders, metrics, logs без prompt content, PASS/FAIL.

| Agent / этап | Конкретные входы | Deliverable | PASS/FAIL gate |
| --- | --- | --- | --- |
| **A0 Model/IR auditor, P0** | C6 `src/llm/servable_initializer.cpp`, `src/llm/language_model/continuous_batching/servable_initializer.cpp`; upstream `src/plugins/intel_gpu/src/plugin/ops/moe.cpp`; Gemma4, Qwen3.6, Ornith checkpoint manifests | `model-capability-matrix` с graph ops, exact quant/IR/drafter artifacts, per-model `OFFLOAD_RATIO` proof | **FAIL** при неподтверждённом OTD/MTP; не маскировать legacy Qwen3 как product success |
| **A1 Baseline/protocol, P1** | C6 tool fixtures, `tests/`, target checkpoints, C1/C2/C4, 8K/32K/64K, KV U4 | A/B resident vs valid ratios 20/40/60/80, plain/MTP/DFlash where eligible; per-model agent loops | **PASS** exact named/required/auto/parallel tool calls, no corrupted output, stable multi-slot; otherwise stop |
| **A2 Native OTD profiler, P2** | Upstream `expert_weight_providers.cpp`, `moe_otd_runtime.cpp/.hpp`, `lru_cache.cpp`; OS I/O and GPU profilers | Reproducible miss/transform/copy/stall breakdown, physical bytes/token, run counters | **PASS** read/transpose/copy reconciliation; no fabricated per-request native counter |
| **A3 Trace + replay, P3** | Per-layer router events from eligible plugin path, separate draft/verify phases, exact checkpoint/quant | Versioned trace schema, deterministic LRU replay vs candidate CLOCK/2Q/routing-aware, C1/C2/C4 | **PASS** exact top-k replay and measured bytes/committed token; no private tokens/paths |
| **A4 Packed-layout, P4** | Actual IR offsets, weight/scales/ZP, numeric reference | Optional prepack/transform prototype, hashes and parity fixtures | **PASS** bit/quant semantics and logits parity, measurable bottleneck improvement |
| **A5 Cache/I/O candidate, P5** | A2+A3 evidence, isolated Intel GPU plugin worktree | Feature-flagged priority queue, bounded prefetch, lease/cancel, optional policy; clean fallback to native | **PASS** target demand never starved, no extra paging, p95/p99 fairness not worse, SSD bytes/token improve or capacity clearly improves |
| **A6 CPU expert lane, P6** | Separate kernel bench on 258V AVX2, Ryzen AVX2, P720 Xeon 6242 AVX-512 VNNI | ISA-gated reference kernels, NUMA-local placement cost model | **PASS** numerical parity and measured end-to-end gain; otherwise keep disabled |
| **A7 Integration, P7** | Only approved candidate commits + pinned C6, model/agent fixtures | 30–60 min soak, C1/C2/C4, long context, fault-injected NVMe/memory, rollback procedure | **FAIL** on protocol regression, OOM/swap storm, wrong logits, slot starvation, speculative state desync |
| **A8 Feasibility, P8** | Qwen3.8 mixed-quant manifest, real memory/IO numbers | Model-specific budget/precision study, no unsupported performance claims | **PASS** only with verified export/kernel support, physical footprint and quality |

### Minimum trace record (proposed, not an existing API)

```json
{
  "run_id": "synthetic-run-id",
  "checkpoint_sha": "sha-placeholder",
  "quant_profile_sha": "sha-placeholder",
  "slot_id": 0,
  "batch_id": 0,
  "round_id": 0,
  "phase": "verify",
  "layer_id": 0,
  "expert_id": 0,
  "projection": "down",
  "route_weight": 0.0,
  "load_reason": "target_demand",
  "residency": "miss",
  "physical_nvme_read_bytes": 0,
  "io_wait_us": 0,
  "proposal_position": 0,
  "accepted_prefix_length": 0,
  "rejected_proposals": 0,
  "committed_output_tokens": 0,
  "kv_u4_unique_bytes": 0
}
```

Данные о фактическом токене/промпте **не записывать**. События draft и verify должны различать `checkpoint_sha`, `phase`, `slot_id` и `round_id`. Агрегация `bytes/committed token` ведётся по реальным storage counters, отдельно prefill и decode; «логический miss» не эквивалентен физическому NVMe read.

## 8. Проверяемые первоисточники

- [OpenVINO 2026.4.0 — expert weight provider API](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/expert_weight_provider.hpp), [native provider and LRU](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/expert_weight_providers.cpp), [LRU policy](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/lru_cache.cpp), [OTD runtime](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_runtime.cpp), [MoE graph gate](https://github.com/openvinotoolkit/openvino/blob/2026.4.0/src/plugins/intel_gpu/src/plugin/ops/moe.cpp).
- [Strata model support](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md), [Qwen3.6 MTP/multi-request](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md), [Qwen3.8 mixed quants](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md), [Intel backend limitations](https://github.com/Niko1221/Strata/blob/main/docs/INTEL.md). **Strata Intel B70/NVIDIA/AMD benchmarks ≠ Arc 140V Windows/UMA results.**
- [Unsloth Qwen3.6 MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), [AEON Qwen3.6/Ornith DFlash](https://huggingface.co/AEON-7/AEON-DFlash-Qwen3.6-35B-A3B): existence does not prove compatible OpenVINO export.
- [Manual original research export](../../../routed-experts-research-original.md), [existing roadmap](06-IMPLEMENTATION-ROADMAP.md), [model/drafter evidence](07-MODEL-AND-DRAFTER-BASELINES.md), [speculative cache study](08-SPECULATIVE-EXPERT-CACHE-RESEARCH.md).

**Decision gate:** архитектурный план v2 разрешён к публикации как **docs-only proposal**. Он **не** утверждает завершённость нового Deep Research, поддержку моделей в pinned OpenVINO, производительность на Arc 140V или реализацию нового cache scheduler.
