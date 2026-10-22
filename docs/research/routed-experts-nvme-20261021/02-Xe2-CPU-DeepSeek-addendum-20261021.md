# Дополнение к исследованию Routed Experts / MoE NVMe-offload

**Дата:** 21 октября 2026 г.  
**Статус:** ADD-ONLY appendix. Основной исследовательский отчёт «Routed Experts / MoE с NVMe-offload в OVMS/OpenVINO на Intel Arc 140V» **не заменяется, не сокращается и не правится**. Этот материал предназначен для добавления после его существующих разделов. C6 / основной runtime GEMMAMONSTER **не изменять**.  
**Предмет:** проверенный KV U4; форматы Xe2 XMX; DeepSeek V4-Flash SSD expert streaming; CPU expert lanes на старых Ryzen, Xeon Scalable 2+ и Core Ultra 7 258V; выбор переносимой математики.

## A. Новый локальный baseline: KV U4, не U8

- **Зафиксированный факт пользовательских A/B:** на Arc 140V **KV U4 многократно испытан, работает стабильно и является лучшим исходным режимом**. U8 **не** должен оставаться рекомендованным baseline, хотя фигурирует в первоначальном отчёте.
- Для новой экспериментальной ветки использовать **KV U4 + существующий scheduler + prefix caching**. U8 — только контрольная точка для отдельного regression/quality сравнения. Не начинать исследование OTD с попытки заново доказать уже установленный выигрыш U4.
- **Важно различать:** KV U4 (хранение K/V), weight INT4 (хранение MoE весов), нативные INT4 XMX операции и MXFP4/FP4 формат DeepSeek. Одинаковое число бит **не означает одинаковую математику**.
- Активный KV оставить в LPDDR. NVMe KV snapshot/hibernation рассматривать независимо, как и в исходном отчёте; отдельное SSD-хранилище KV не означает произвольные NVMe reads на каждом decode token.
- Регрессионные проверки U4: контекст 1K/8K/32K и рабочие длинные контексты, batch/concurrency 1/2/4, prefix hit/miss, attention accuracy, tool-call reliability, RSS/working set, TTFT/p95 TPOT.

## B. Arc 140V: INT4, INT8, BF16 или FP16? Аппаратная проверка

Intel официально описывает для **Xe2 / Lunar Lake** матричные движки **XMX FP16, BF16, INT8, INT4, INT2**. В презентации Xe2, *Key Peak Metrics for 2nd Gen Xe-core*, указаны следующие теоретические максимумы **на Xe-core за такт**:

| Операция | Peak ops/clock/Xe-core | Отношение к XMX FP16/BF16 |
|---|---:|---:|
| XMX FP16 / BF16 | 2,048 | 1× |
| XMX INT8 | 4,096 | 2× |
| XMX INT4 / INT2 | 8,192 | 4× |
| Vector FP32 | 256 | не сравнивать как эквивалентный kernel |

**Источник:** [Intel Xe2 and Lunar Lake GPU Deep Dive, слайд 9](https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf); [Intel Xe GPU Architecture](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html) подтверждает наличие XMX/DPAS у Xe2-LPG и общую память.

**Интерпретация:** физически Xe2 предпочитает матричную низкую разрядность **при идеально загруженном XMX**, но **настоящий победитель на Arc 140V пока не установлен**. Пики не гарантируют TPS для MoE GEMV при `M=1`: влияет occupancy, shape/tiling, pack/unpack, group scales, dtype casts и количество перемещаемых весов. **MXFP4 E2M1 с E8M0 scales не равен integer INT4;** потребуется правильный decoding/scaling, а это может устранить теоретические 4×.

**Обязательный precision sweep:**

| Weight/storage | Compute path | Что узнать |
|---|---|---|
| INT4 (OpenVINO/NNCF symmetric/asymmetric) | XMX INT4 или распаковка → INT8/FP16/BF16 (фактический dispatch выяснить) | Действительно ли используется packed INT4 DPAS? |
| INT8 weights | XMX INT8 | Меньше overhead, больше bytes, иногда быстрее малых матриц |
| FP16 weights | XMX FP16 | Контрольная точка FP |
| BF16 weights | XMX BF16 | Контрольная точка BF и устойчивость динамического диапазона |
| MXFP4 experts (DeepSeek) | нативный MXFP4 decode → пригодный DPAS формат | Время E2M1 decode, E8M0 scales, формат GPU tiles |
| Q4_K / NF4 / EXL3 | format-specific decode + INT/FP kernel | Издержки перекодирования; lossless/tolerance tests |

**Матрицы/тесты:** `M={1,2,8,32,128}`; `N,K` взять из реальных экспертов Qwen и DeepSeek; `top_k={2,4,6,8}` в пределах архитектурной корректности; FP32 accumulation там, где её требует эталон; раздельно `kernel-only`, `dequant`, `route+gather`, `expert+merge`, `full TPOT`. Записать фактическое kernel implementation name / perf counters и проверить использование DPAS/XMX, а не верить названию precision в конфигурации.

## C. DeepSeek V4-Flash: новые доноры SSD-routing и математики

### C.1. `ssd-moe/deepseek-v4-flash-mlx` — готовый SSD-streaming дизайн

**Репозиторий:** https://github.com/ssd-moe/deepseek-v4-flash-mlx

- Исследовательский MLX runtime для DeepSeek V4-Flash на Apple Silicon с **48 GB unified memory**.
- Dense/attention часть resident в более высокой точности (8 bit); routed experts **MXFP4 остаются на SSD**.
- Active top-k expert records читаются по offsets (`pread`/`preadv`) с параллельным `ThreadPoolExecutor`, затем перемещаются в fixed-size device LRU. Async batch reads перекрывают вычисление shared expert.
- README заявляет **~4.5–5 tok/s** decode на 48-GB Mac. Это авторское измерение, **не** benchmark Arc 140V, и 48 GB больше имеющихся 32 GB.
- `CAP` задаёт resident expert slots; tune against real free RAM. Не делать вывод о достижимых 5 tok/s на 140V из этих данных.

**Реально прочитанные файлы:**

- [`oracle/offload_cache_v4.py`](https://github.com/ssd-moe/deepseek-v4-flash-mlx/blob/main/oracle/offload_cache_v4.py): `TwoTierExpertCache`, routed-only LRU, pinned shared tensors, guards против eviction non-routed; отделение resident data от structural accounting.
- [`oracle/pread_loader.py`](https://github.com/ssd-moe/deepseek-v4-flash-mlx/blob/main/oracle/pread_loader.py): offsets safetensors; `os.preadv` / `os.pread`, `ThreadPoolExecutor`, `load_batch_async`, `load_batch_join`; I/O parallel, MLX tensor write на главном потоке.
- [`oracle/v4_fast.py`](https://github.com/ssd-moe/deepseek-v4-flash-mlx/blob/main/oracle/v4_fast.py): asynchronous expert load while shared work executes; `mx.quantized_matmul`, `mode="mxfp4"`, `group_size=32`, `bits=4`.

**Что переносить:** expert-only LRU guards, offset-indexed contiguous loading, async I/O futures, join-before-consume correctness, explicit RAM cap. **Что НЕ переносить напрямую:** `mlx.core`, Apple Metal storage and tensor kernels; эти классы надо переосмыслить поверх существующего `IExpertWeightProvider` OpenVINO.

### C.2. `pjordanandrsn/experts4bit-qlora` — аккуратная математика DeepSeek V4

**Файл:** https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/DEEPSEEK-V4.md

- Описаны storage split **routed MXFP4 vs dense FP8**, aligned NVMe arena и resident kernel semantics.
- Для MXFP4 scales `F8_E8M0` надо подавать в `ldexp` как **raw exponent byte**; для FP8 dense scales требуется multiplier вида `2^(byte-127)`. Ошибка конвертации не обязательно вызывает краш, но искажает вывод.
- Проверять nibble ordering, shape, key mapping, grouped projection vs dense projection.
- **Критичное fidelity в DeepSeek V4:** `gate=clamp_max(gate,L)`, `up=clamp(up,-L,L)`, `silu(gate)*up`; вычисления activation в FP32; routed gate weight умножается **перед** `down_proj`, а не после, если так предписывает эталон. Различие мест умножения способно изменить результат вследствие quantization/casts.
- Если backend не реализовал MTP, это нужно фиксировать как явное функциональное ограничение, а не молча считать его поддержанным.

Официальный эталон модели: [deepseek-ai/DeepSeek-V4-Flash, inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py). Именно с ним сравнивать math, а не только с любым сторонним runtime.

**Важно:** оптимизированные DeepSeek kernels **model-/format-specific**. Маршрутизацию и store можно сделать общими; epilogue, clamping, dequantize и scale conventions обязаны оставаться model-specific, иначе получим быстрый математически неверный вывод.

### C.3. `glm-flash-lite` — CPU+GPU lane уже доказан на AVX2

**Основной документ:** https://github.com/sybil-solutions/glm-flash-lite/blob/main/docs/how-it-works.md

На машине авторов (EPYC 7443P, **22 AVX2 workers**, RTX 3090 с собственной VRAM, восемь каналов DDR4, четыре NVMe) часть выбранных экспертов из RAM считается прямо CPU, а остальные — GPU. Измерено:

- Примерно **150 experts/token** в CPU lane (около 41–45% выбранных experts в приведённом C1).
- При добавлении CPU lane (совместно с prefetch) авторы получили **8.28 → 15.30 tok/s** decode на 55-GiB-cap режиме; **это не изолированный CPU-only эффект** и не результат на 140V.
- CPU lane в этих тестах *не bit-exact*: mean KL примерно 0.0047, top-1 agreement 0.986 для одного заявленного сравнения. Следовательно, порт должен пройти самостоятельный quality gate.
- Prefill CPU lane **не использует**: выгодная decode-политика не обязана выигрывать prefill.
- Есть оптимизированный experimental kernel `GLM53_NV_CPU_KERN=1`, поддерживающий быстрый pair decode.

**Реально прочитанный код:**

- [`kernels/nv2/ft_core.h`](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_core.h): AVX2 `_mm256_shuffle_epi8`, `_mm256_madd_epi16`, FMA, fp32 activation, EXL3 trellis MUL1 codebook, SwiGLU clamp, worker pool, reference double path.
- [`kernels/nv2/ft_n135.h`](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_n135.h): p16 pair decoding, batch=1-oriented register accumulators, tile-major row traversal.
- [`kernels/nv2/nv2_host.cpp`](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/nv2_host.cpp): controller / CPU worker / NVMe reader engine.

**Портировать идею:** expert assignment `GPU_hit | GPU_admit | CPU_resident | NVMe_to_CPU | NVMe_to_GPU`, отдельное CPU worker pool, partial-sum merge, completion flag, immutable source. **Не копировать слепо математику:** `ft_core` специализируется на **EXL3 MUL1 K=3 trellis** и GLM activation, это не универсальный Qwen/DeepSeek/OpenVINO INT4 kernel. Требуются адаптер квантования и правильный epilogue.

### C.4. CPU kernels в `llama.cpp` / AMD ZenDNN / oneDNN

- [`ggml-org/llama.cpp` CPU/GGML](https://github.com/ggml-org/llama.cpp): проверенные SIMD quantized GEMV/GEMM и MoE-операция `MUL_MAT_ID`; полезен прежде всего как generic reference и microbenchmark donor.
- [`llama.cpp ZenDNN backend`](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/ZenDNN.md): ускоряет именно `MUL_MAT` **и** `MUL_MAT_ID`; заявленные FP32, BF16 (Zen4/Zen5 лучше), Q8_0. На старых Ryzen нельзя автоматически рассчитывать на нативный BF16.
- [OpenVINO CPU plugin](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html): oneDNN-based runtime, native hardware ISA dispatch; Intel AMX BF16/INT8 на Xeon 4+; `BF16` на CPU без нативной поддержки может быть software emulation и **медленнее FP32**.
- **Лицензирование:** просмотренные `LICENSE` у `glm-flash-lite`, `ssd-moe/deepseek-v4-flash-mlx`, `experts4bit-qlora`, `Strata` и `llama.cpp` — MIT на текущих main/master. При заимствовании файла соблюсти copyright / NOTICE и проверить лицензии внешних зависимостей и модели. Лицензия одной репы не покрывает автоматически model weights.

## D. CPU classes: реальные ISA, бюджет памяти и выбор expert lane

| Класс | ISA и формат, который стоит тестировать первым | Возможная роль | Особые риски |
|---|---|---|---|
| Старые Ryzen Zen1/Zen2/Zen3, включая Zen3+ 6900HX | AVX2 + FMA, packed Q4/Q8, **FP32 accumulation**; без нативного BF16/AMX на этих поколениях | CPU standalone и гибридная expert lane с discrete GPU | DDR bandwidth и число каналов; в Zen1 AVX2 исполняется иначе, чем в Zen2/Zen3; Q4 decode overhead |
| **P720: 2× Xeon Gold 6242** (Cascade Lake, 2nd Gen Scalable), 384 GiB | **AVX-512 VNNI / AVX2**, INT8 dot и packed Q4 decode; **без AVX512_BF16 и без AMX** | Самый интересный локальный стенд для больших resident MoE и AVX512 CPU lane | 2× NUMA: remote DRAM, недооценка памяти, `numactl`/thread pinning, socket-crossing reduction |
| Xeon Scalable 3rd Gen Ice Lake | AVX-512 VNNI; не считать AVX512_BF16 универсальной характеристикой всей 3-й генерации | Большой CPU resident pool, NUMA-aware MoE | NUMA и bandwidth; AMX нет |
| Xeon Scalable 3rd Gen Cooper Lake | AVX-512 VNNI **+ AVX512_BF16**, без AMX | BF16 matmul при подходящих размерах | Нужно точное SKU/CPUID |
| Xeon Scalable 4th+ Sapphire / Emerald и последующие, в зависимости от SKU | AVX-512 VNNI + **AMX INT8/BF16** (для конкретного поколения уточнить ISA, ОС и доступность AMX); AVX512_BF16 | Ускорение batched expert GEMM / prefill и INT8 decode | Для M=1 AMX не всегда выигрывает над AVX512 GEMV; tile setup и dequant overhead |
| **Core Ultra 7 258V** CPU (Lunar Lake: **4P+4E, 8 потоков**) | Intel публично подтверждает **AVX2**; дополнительные VNNI-возможности проверять CPUID. **Нет основания ожидать AMX или AVX-512 BF16** | Небольшая CPU lane для GPU misses, router/prefetch bookkeeping | **Одна LPDDR и общий power/thermal budget с Arc 140V**, нехватка свободных P-ядер, GPU downclock |

Источники: [Intel Xeon AVX-512 generational table](https://www.intel.com/content/www/us/en/support/articles/000058341/processors/intel-xeon-processors.html); [Intel 4th Gen Xeon tuning guide, VNNI/BF16/AMX](https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-guide-for-ai-on-the-4th-generation.html); [Xeon 6242 specifications](https://www.intel.com/content/www/us/en/products/sku/192440/intel-xeon-gold-6242-processor-22m-cache-2-80-ghz/specifications.html); [Core Ultra 7 258V specifications](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html).

### Коррекция к первоначальному выводу про Arc 140V и CPU lane

Общая LPDDR у GPU и CPU — **аргумент против слепого копирования результатов EPYC+RTX**, но **не доказательство отрицательной эффективности CPU lane**. Даже при общей RAM CPU может выигрывать за счёт одновременной работы со GPU, устранения device staging/slot thrashing и уменьшения ожидания resident miss; может проигрывать от bandwidth contention, CPU↔GPU sync и падения частоты GPU. Поэтому статус **«не включать в первый production MVP, но проверить экспериментальным gate»**.

Особенно важно: OpenVINO `compile_model(..., "CPU")` **не означает**, что GPU plugin автоматически отправляет selected experts CPU. Гибридному lane нужны runtime scheduling, общий ID mapping, engine-specific CPU math, partial sums и правильная синхронизация. `HETERO:GPU,CPU` не следует считать готовой реализацией token-level conditional routing.

Для **P720** проект может оказаться даже более перспективным, чем для 140V: **384 GiB** позволят держать многие крупные MoE resident без NVMe miss на decode, а два Xeon дадут значимый CPU lane. Нужны NUMA-local expert shards, pin workers+DRAM к одному сокету, сравнить 1-socket/2-socket, **не** разносить один `M=1` вызов через UPI без необходимости.

## E. Проект CPU expert planner: переносим общую схему, не несовместимые kernels

Минимальный самостоятельный experimental интерфейс, не часть принятого API:

```text
Router → (layer, expert, weight, token/group)
          |
       residency + latency-aware planner
          |                         |
          v                         v
       GPU expert jobs          CPU expert jobs
       hot / admit             resident or NVMe-ready
          |                         |
          v                         v
       GPU partial sums        CPU partial sums (FP32)
          \                         /
           \--- weighted merge -----/
                      |
                    next op
```

**Стоимостная функция, измеряемая на каждой машине:**

```text
T_gpu_path = T_cache_miss + T_io + T_stage/dequant + T_gpu_kernel
T_cpu_path = T_io_to_host_if_needed + T_cpu_dequant + T_cpu_kernel + T_merge

CPU route candidate iff estimated completion of hybrid layer
    < estimated completion of pure GPU layer
AND physical LPDDR / NUMA / power headroom safe.
```

Замерять не только изолированный `T_cpu_kernel`, но и **critical path**: слой закончен, когда готова **последняя** partial sum. При тяжёлом CPU tail GPU простаивает, и общий TPS падает даже если CPU kernel выглядит быстрым.

**Минимальный portable math contract:**

```text
ExpertMath(model, expert_quant, shape)
 ├─ decode weights / scales (INT4, EXL3, MXFP4, GGUF)
 ├─ W1/W3 gated matvec or matmul
 ├─ model-specific epilogue (SwiGLU, clamps, casts, weight position)
 ├─ W2 projection
 └─ FP32 partial output + correctness metadata
```

`GLM ft_core` применять **только** к совместимым EXL3; `DeepSeek` math из официального эталона кодировать отдельно; `llama.cpp` GGML `MUL_MAT_ID` и oneDNN `brgemm` рассматривать как наиболее переносимые generic SIMD backends. В OpenVINO GPU OTD сначала интегрировать только **observability / prototype lane**, не нарушая C6 baseline.

## F. Конкретная исследовательская матрица, разделённая по оборудованию

**Этап 1: ISA/capability probe.** Для всех CPU сохранить `lscpu` или Windows CPUID dump, NUMA, каналы DRAM, SKU и freq. OpenVINO CPU capabilities:

```python
import openvino as ov
core = ov.Core()
print(core.get_property("CPU", "OPTIMIZATION_CAPABILITIES"))
```

Обязательно приложить реальный вывод: `BF16` в capability list и actual oneDNN kernel — разные уровни свидетельства; для GPU similarly capture compiled kernel type/implementation.

**Этап 2: kernels (одинаковые реальные expert shapes/weights).**

| Платформа | Сравниваемые lanes / форматы |
|---|---|
| Ryzen AVX2 | OpenVINO CPU/oneDNN vs GGML AVX2 `MUL_MAT_ID` vs специализированный kernel при подходящем quant; FP32/Q8/Q4 |
| P720 Xeon 6242 | 1-socket AVX2 vs AVX512 VNNI, INT8 vs Q4, 2-socket NUMA-aware, RAM resident vs NVMe-backed |
| Xeon 3/4/5+ | VNNI vs BF16(native CPUID) vs AMX INT8/BF16, `M=1` vs batched M; measure actual CPU execution path |
| 258V CPU+Arc 140V | GPU-only XMX INT4/INT8/BF16/FP16; CPU-only 1–4 P cores; hybrid 1–4 P with/without E; shared LPDDR pressure |

**Этап 3: async hybrid (только если kernel proof пройден).**

- `GPU-only`, `CPU-only` (selected experts), `HYBRID`.
- CPU assignments `0/1/2/4/...` experts/layer до точки насыщения.
- `M=1` (decode), `M=8/32/128` (prefill/batch) отдельно.
- `memory budget fixed`, U4 KV fixed, same model, same expert storage, same prompts, same exact selected IDs and weights.
- SSD buffered vs O_DIRECT где актуально; VM page-cache cold/warm разнести.
- На 258V обязательно измерить **GPU frequency**, CPU package power, RAPL/thermal throttling и LPDDR contention: иначе ложный «CPU помогает» на коротком прогоне.
- На Xeon обязательно `numactl --cpunodebind=... --membind=...` или equivalent NUMA pin и второй arm cross-node.

**KPIs:** TTFT, decode TPS, TPOT p50/p95/p99; CPU completion tail per layer; GPU idle/wait; expert/cache hits, misses, offload bytes/token, **actual SSD MB/token**, resident RAM, GPU memory allocations, kernel effective GB/s and MAC/s, clock/power/temperature; output KL/top-1 agreement vs official reference.

**Acceptance gates (предложение):**

1. `CORRECTNESS`: same selected expert IDs, correct packing/nibble/scale interpretation; model-native activation/weight positions; deterministic reference vectors. CPU lane may be approximately but must pass pre-agreed KL/top-1 tolerance, with separate rare-case regression corpus.
2. `LATENCY`: hybrid p95 TPOT improves >=10% vs same-run GPU-only OTD (illustrative project threshold; agree before running), no substantial TTFT degradation.
3. `MEMORY`: 32GB 258V keeps safety headroom, **не уходит в swap/pagefile**.
4. `POWER`: long-soak result persists, no GPU thermal collapse.
5. `IO`: SSD MB/token does not increase materially simply because CPU lane fetches the same expert twice.
6. `PORTABILITY`: Windows 11 Arc140V vs Linux test separately; Ryzen AVX2 binary must not execute AVX512; Cascade Lake path no illegal BF16/AMX.
7. `INTEGRATION`: no changes to pinned C6 until isolated prototype proven and independently reviewed; correct model-specific parser/tool-calling unaffected.

## G. Решение о порядке работ

**Обновлённая очередь без отмены старого отчёта:**

1. **Принять KV U4 как measured baseline**, U8 убрать из роли default (сохранить как контрольное плечо, не удалять прежний текст).
2. На **Arc 140V** проверить precision dispatch `XMX INT4 vs INT8 vs FP16/BF16`, отдельно `weights` и `KV`; найти реальные winner cases по batch/shape.
3. Native OpenVINO `OFFLOAD_RATIO` и `WEIGHTS_PATH` на отдельном runtime; измерить misses/bytes/token.
4. Сравнить три независимых SSD-стека: **OpenVINO OTD**, **Strata hotset**, **DeepSeek V4 MLX pread+LRU** плюс `glm-flash-lite` async/direct-I/O. Брать переносимые алгоритмы, не CUDA/MLX kernel ABI.
5. Исследовать **CPU expert math**: generic GGML MoE, oneDNN и GLM AVX2, точный MXFP4 DeepSeek decoder. Сначала microbench CPU-only на Ryzen и P720.
6. На **258V** испытать hybrid CPU/GPU только после microbench и корректности, не включать его по умолчанию.
7. P720 с 384 GiB — отдельная ветка **RAM-resident / NUMA-aware CPU experts**; у неё другие пределы, чем у 32GB Arc140V.
8. Потом optional async prefetch / admission в OpenVINO GPU plugin, строго в кандидатном worktree, с регрессионными acceptance-gates.

## H. Проверяемые публичные источники

- Intel Xe2: https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf
- Intel GPU architecture: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html
- Intel 258V: https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html
- Intel 2nd/3rd Xeon AVX512: https://www.intel.com/content/www/us/en/support/articles/000058341/processors/intel-xeon-processors.html
- Intel 4th Xeon VNNI/BF16/AMX: https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-guide-for-ai-on-the-4th-generation.html
- Intel Xeon 6242: https://www.intel.com/content/www/us/en/products/sku/192440/intel-xeon-gold-6242-processor-22m-cache-2-80-ghz/specifications.html
- OpenVINO CPU: https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html
- OpenVINO native OTD: https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe
- OpenVINO/OVMS offload issue: https://github.com/openvinotoolkit/model_server/issues/4580
- Strata: https://github.com/Niko1221/Strata
- GLM tiering: https://github.com/sybil-solutions/glm-flash-lite/blob/main/docs/how-it-works.md
- GLM CPU core: https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_core.h
- GLM optimized CPU: https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_n135.h
- DeepSeek MLX SSD: https://github.com/ssd-moe/deepseek-v4-flash-mlx
- DeepSeek `pread`: https://github.com/ssd-moe/deepseek-v4-flash-mlx/blob/main/oracle/pread_loader.py
- DeepSeek LRU: https://github.com/ssd-moe/deepseek-v4-flash-mlx/blob/main/oracle/offload_cache_v4.py
- DeepSeek MXFP4 details: https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/DEEPSEEK-V4.md
- DeepSeek official reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py
- llama.cpp ZenDNN MoE: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/ZenDNN.md

**Итог:** перенос CPU expert lane технически возможен как отдельный кандидатный механизм, особенно на AVX2 Ryzen и AVX512 Xeon. Его выгода на Arc 140V **не предрешена** ни в плюс, ни в минус, несмотря на общую память. Для 140V сначала подтверждаем XMX precision winner, native OTD и SSD traffic, а затем решаем, когда CPU полезнее GPU. Все предыдущие выводы исследования остаются на месте; это исключительно новая доказательная надстройка.