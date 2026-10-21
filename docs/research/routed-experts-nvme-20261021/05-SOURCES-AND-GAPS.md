# Source manifest, fidelity notes and unresolved inputs

**Date:** 2026-10-21.

## Available source documents published here

1. Earlier standalone document `ARC-140V-research.md`, dated 2026-10-06 (source 13,944 UTF-8 bytes). Public copy at `01-ARC-140V-research-20261006.md`; exactly one machine example path `<LOCAL_MODEL_DIR>` replaced with `<MODEL_REPOSITORY>` to avoid publishing a local path. No technical recommendations were changed. That older note is *historical*, not a final 2026-10-21 report.
2. Standalone document `Routed_Experts_Xe2_CPU_DeepSeek_addendum_20261021.md`, dated 2026-10-21 (original content preserved in `02-...`).
3. Strata audit and OVMS integration notes newly authored during this publication; explicitly marked supplementary analyses.

## Not included / cannot claim uploaded

The main comprehensive Deep Research text titled **«Интеграция Routed Experts и механизмов GLM-Flash-Lite с NVMe в ОВМС на Intel Arc 140V»**, described as approximately 52k characters, was displayed inside a Deep Research UI result, not attached as a downloadable text artifact in this working session. The exact prose is not available for lossless export from the present context. **It is not in this repository commit**. DO NOT label our synthesis as that primary report; do not recreate invented quotes or claim full-file hash equality.

A Strata-specific addition was requested during Deep Research; the exact final generated research output is not available as a separate verified file. The independent Strata source audit here is a new additive document, not a purported exact historical export.

## Research status

- **User-verified hardware evidence:** repeated KV U4 success on Arc 140V.
- **Source verified, no local run in this packet:** Strata Linux SYCL restrictions, GLM AVX2 CPU lane, DeepSeek MLX read/cache design.
- **Hypothesis / test needed:** precision winner on 258V Xe2, end-to-end CPU/GPU hybrid win on UMA, Xeon P720 CPU tier speed, NVMe miss distribution.
- **Historical finding requiring reconfirmation:** exact native OpenVINO/OVMS disk-offload parameter names and model compatibility can vary by release; identify the binary and linked plugin revision before benchmarking.

## Sources and licensing

- [OpenVINO](https://github.com/openvinotoolkit/openvino) and [OVMS](https://github.com/openvinotoolkit/model_server)
- [GLM-Flash-Lite](https://github.com/sybil-solutions/glm-flash-lite) (MIT)
- [Strata](https://github.com/Niko1221/Strata) (MIT)
- [DeepSeek V4-Flash MLX SSD streamer](https://github.com/ssd-moe/deepseek-v4-flash-mlx) (MIT)
- [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) (MIT)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT)

Research ideas are not equivalent to permission to transplant all implementation code without reviewing the full upstream license and dependencies.
