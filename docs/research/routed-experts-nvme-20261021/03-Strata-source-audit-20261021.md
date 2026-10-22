# Strata: источниковый аудит подходов к Routed Experts / NVMe для GEMMAMONSTER

**Дата:** 2026-10-21  
**Уровень:** анализ открытых источников, без локального бенчмарка на Arc 140V  
**Репозиторий:** [Niko1221/Strata](https://github.com/Niko1221/Strata)  
**Объект:** перенос *идей*, а не прямое включение Strata runtime в OVMS.

## 1. Статус Intel Arc SYCL

[docs/INTEL_ARC.md](https://github.com/Niko1221/Strata/blob/main/docs/INTEL_ARC.md) и [docs/INTEL.md](https://github.com/Niko1221/Strata/blob/main/docs/INTEL.md) на доступной версии описывают экспериментальный самостоятельный движок `sycl/`, портированный с CUDA через oneAPI. Он **Linux-only**, отключён по умолчанию в других сборках, native Windows Intel path не существует. Порт зависит от oneAPI/oneMKL и корректных драйверов `xe` / `i915`. Это не тот же самый backend, что SYCL в llama.cpp.

Указанные в документации результаты — **не результаты Arc 140V**:

| Устройство | Описание результата в источнике | Следствие |
| --- | --- | --- |
| Arc Pro B70 32 GB (дискретная) | Intel port автора: Coder IQ1_M, 70–78 ток/с; измерен большой контекст | Полезен для проверки корректности Xe2-kernels, не для UMA-бюджета |
| Arc A750 8 GB (дискретная) | Поддерживается в лабораторных тестах maintainer | Отличается архитектурой и драйверным стеком |
| Arc B580 в WSL2 | Community-run ~21 ток/с и device-loss caveat | Нельзя считать Windows-поддержкой |
| **Arc 140V 32 GiB UMA** | **Нет подтверждённого native Windows/SYCL теста** | Нужен новый planner и hardware gate |

Документация прямо называет integrated Arc **untested** и указывает на отсутствие planning для общей памяти. Поэтому фраза «Strata поддерживает Intel» не равна «Strata запускается на Windows + Arc 140V».

## 2. Как устроены эксперты и память

Strata делает **expert residency/staging**, где особенно полезны четыре приёма:

- residency table с точным указанием расположения routed expert;
- отделение горячих экспертов от холодных, измеряемый working set;
- request ring / staging / перекрытие disk→host→device с вычислениями;
- специализированные quantized kernels с validation/parity tests.

Однако встраивать disk→GPU→CPU копирования 1:1 в 258V неверно: LPDDR общая, нет самостоятельной VRAM с PCIe-копированием. Стоимость определяется memory placement, device allocation, копиями и синхронизацией, а не пропускной способностью PCIe.

### Кандидат для OVMS: единый экспертный backing store

```text
NVMe packed expert store (immutable, checksummed)
                |
        async pread + scheduler
                |
      shared-memory staging arena
             /          \
     GPU/XMX resident   CPU SIMD lane (optional)
             \          /
         sum of routed expert outputs
```

Этот рисунок — **предлагаемая архитектура**, не утверждение о существующей реализации в OVMS.

## 3. Precision: KV U4 отдельно от expert arithmetic

Отдельные оси:
1. **KV cache:** пользовательский known-good U4 на Arc 140V (не подменять U8).
2. **Expert storage:** MXFP4, EXL3, GGUF I-quant, NF4, INT4/INT8, FP8 — разные кодеки; нельзя просто интерпретировать bytes одинаково.
3. **Compute:** XMX INT4 / INT8 / BF16 / FP16, CPU AVX2/VNNI/AMX, mixed precision accumulation.
4. **Correctness:** сравнение reference logits, паритет на реальных весах, отсутствие NaN и иных регрессий в tool loops.

Xe2 theoretical peak — лишь верхняя граница, особенно при batch=1 GEMV.

## 4. Prefetch и минимизация SSD latency

Переносимые компоненты: layer-ahead hints, asynchronous requests, SSD read coalescing, LRU/CLOCK admission, prefill и decode как отдельные рабочие режимы. На каждом тесте отслеживать read amplification и долю hit/miss, p50/p95 stall time, фактический объём resident experts и degraded behavior при переполнении.

Маршрутизатор **не имеет права менять набор реальных top-k экспертов для экономии SSD** без отдельного quality-gate. Prefetch hints — не изменение математики модели.

## 5. Обязательные проверки Strata→OVMS

| Gate | Проверка | Условие |
| --- | --- | --- |
| S0 | exact commit, model identity, hardware/runtime versions | Записать до тестов |
| S1 | kernel numerical parity | Включить real weight fixtures |
| S2 | memory safety на UMA | Не допускать paging storm и OOM |
| S3 | KV U4 + prefix cache | Сохранить known-good protocol / tool calling |
| S4 | routed cache and SSD instrumentation | Hit/miss и byte accounting |
| S5 | CPU/GPU lane comparison | t/s, TTFT, stalls, power |
| S6 | long-context + agentic | Повторяемые 32K/64K/128K и tool loops |

При невозможности S1–S3 кандидат остаётся research-only.

## 6. Ограничения переноса

- В Intel SYCL-портe документированы device-side особенности, `SYCL_CACHE_PERSISTENT=0`, условия `STRATA_VERIFY_NO_HOST=1`, необходимость осторожного memory planning.
- GPU kernels, конкретные CUDA/SYCL API, scheduler и quantized formats нельзя копировать в OVMS без адаптации к OpenVINO GenAI.
- Обязательно проверить MIT-лицензию upstream, attribution и compatibility с кодовой базой OVMS при переносе кода; архитектурные идеи можно исследовать независимо.
- **Не менять pinned C6 и staged runtime.** Работу проводить в отдельной research/implementation ветке после acceptance gate.

## 7. Исходники

- [Strata main repo](https://github.com/Niko1221/Strata)
- [Strata Intel Arc status and restrictions](https://github.com/Niko1221/Strata/blob/main/docs/INTEL_ARC.md)
- [Strata detailed SYCL design and benchmarks](https://github.com/Niko1221/Strata/blob/main/docs/INTEL.md)
- [GLM-Flash-Lite architectural reference](https://github.com/sybil-solutions/glm-flash-lite/blob/main/docs/how-it-works.md)
- [CPU tier AVX2 optimized code, EXL3](https://github.com/sybil-solutions/glm-flash-lite/blob/main/kernels/nv2/ft_n135.h)
