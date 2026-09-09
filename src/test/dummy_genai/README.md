# Dummy Cyclic LLM / VLM

Minimal models that always generate the same phrase in an infinite loop:

> **"For the night is dark and full of terrors..."**

This catalog (`src/test/dummy_genai`) currently hosts the *build scripts* for a
dummy LLM and a dummy VLM, and is intentionally named generically so future
dummy GenAI model types (e.g. image generation, speech-to-text) can be added
alongside them.

The built models are published on HuggingFace Hub, not committed to this repo:
- LLM: [`mzeglars/dummy-cyclic-gpt2-ov`](https://huggingface.co/mzeglars/dummy-cyclic-gpt2-ov)
- VLM: [`mzeglars/dummy-cyclic-llava-ov`](https://huggingface.co/mzeglars/dummy-cyclic-llava-ov)

OVMS tests pull them on demand via `prepare_llm_models.sh` /
`windows_prepare_llm_models.bat` into `src/test/llm_testing/mzeglars/...`, the
same convention used for every other real model in that directory (see
"Migration status" below).

Two variants are provided:

| Variant | Architecture | Pipeline types |
|---------|-------------|----------------|
| **LLM** | `GPT2LMHeadModel` | `LM`, `LM_CB` |
| **VLM** | `LlavaForConditionalGeneration` (LLaMA text model + tiny CLIP) | `VLM`, `VLM_CB` |

Both are legitimate HuggingFace models convertible to OpenVINO IR via
`optimum-cli`/`optimum-intel`. Their purpose is to act as **fast,
dependency-free drop-ins** during OVMS integration tests that require a real
LLM or VLM pipeline but do not care about the content of the output.

---

## Quick start

All commands below are run from inside this directory (`src/test/dummy_genai`).

```bash
# Minimal — builds everything with defaults (fp16, infinite cycling)
./build.sh
./build_vlm.sh

# With all parameters shown
./build.sh \
    --weight-format fp16 \       # fp16 (default) | fp32 | int8  (int4 unsupported on NPU)
    --sequence "Winter is coming..." \  # custom phrase; every token must be unique
    --skip-venv \                # reuse existing venv / activated environment
    --skip-export \              # stop after HF model creation, skip OV IR export
    --skip-verify                # skip post-build verification

./build_vlm.sh \
    --weight-format fp16 \
    --sequence "Winter is coming..." \
    --skip-venv \
    --skip-export
```

Both scripts share the same `venv/` and `requirements.txt`. By default each
script creates its own `venv/` here and installs `requirements.txt` into it.
If you pass `--skip-venv` to reuse your own already-active Python environment,
install the dependencies into it yourself first:

```bash
pip install -r requirements.txt
```

### LLM options (`build.sh`)

| Flag | Default | Description |
|------|---------|-------------|
| `--weight-format` | `fp16` | OV weight precision: `fp16`, `fp32`, `int8` (`int4` unsupported on NPU) |

| `--sequence "…"` | `For the night is dark and full of terrors...` | Override the cyclic phrase (every GPT-2 token must be unique!) |
| `--skip-venv` | off | Use the current Python env instead of creating a venv |
| `--skip-export` | off | Stop after creating the HF model; skip OV export |
| `--skip-verify` | off | Skip the verification step |

### VLM options (`build_vlm.sh`)

| Flag | Default | Description |
|------|---------|-------------|
| `--weight-format` | `fp16` | Same as LLM |

| `--sequence "…"` | `For the night is dark and full of terrors...` | Same as LLM |
| `--skip-venv` | off | Same as LLM |
| `--skip-export` | off | Same as LLM |

---

## How it works

Both models use the same core trick: transformer blocks are made identity by
zeroing all attention and MLP weights, so every position's hidden state equals
its own token embedding. A sparse `lm_head` then maps each sequence token's
embedding to the next token in the cycle.

### LLM weight design (GPT-2)

| Component | Value | Effect |
|-----------|-------|--------|
| `wte` | Sequence token *t_i* → **e**_i (standard basis); all others → **e**_{N} | Each token maps to a unique orthogonal direction |
| `wpe` | All zeros | Positions do not corrupt the token embedding |
| Attention + MLP | All zeros | Each block is a pure residual (identity) |
| `ln_f` | weight=1, bias=0 | LayerNorm scales **e**_i by `ln_scale ≈ √(H−1)` |
| `lm_head` | `W[t_{i+1}, i] = 100 / ln_scale` | logit ≈ 100 for the correct next token, ≈ 0 for all others |

### VLM weight design (LLaVA + LLaMA)

The LLaMA text backbone uses RMSNorm, which gives **exact zeros** for
off-diagonal components (cleaner than LayerNorm):

| Component | Value | Effect |
|-----------|-------|--------|
| `embed_tokens` | Same basis-vector scheme as LLM | — |
| *(no position table)* | RoPE is inside attention; zeroed attention → RoPE irrelevant | — |
| Attention + MLP | All zeros | Identity blocks |
| RMSNorm | weight=1 | Scales **e**_i by `√H`; off-diagonal = exactly 0 |
| `lm_head` | `W[t_{i+1}, i] = 100 / √H` | logit = 100 for next token, **exactly** 0 for all others |

The CLIP vision encoder and multimodal projector keep their random default
weights. Because the LM attention is zeroed, image tokens at early positions
never influence the hidden state of the last text token — the cyclic generation
is completely image-agnostic.

### Output files

Build output stays local and gitignored (see `.gitignore`) - it's only an
intermediate step for regenerating the models before re-uploading to HF via
`upload_to_hf.py`, not something OVMS tests read from directly:

```
src/test/dummy_genai/
├── hf_model/          ← LLM HuggingFace model
├── ov_model/          ← LLM OpenVINO IR (stateful KV-cache)
├── vlm_hf_model/      ← VLM HuggingFace model (LLaVA)
└── vlm_ov_model/      ← VLM OpenVINO IR
```

---

## Manual steps

Install the dependencies first (into a venv or your active environment):

```bash
pip install -r requirements.txt
```

```bash
# LLM
python3 create_model.py    --output-dir hf_model
python3 export_to_ov.py    --hf-model-dir hf_model --ov-model-dir ov_model
python3 verify_model.py

# VLM
python3 create_vlm_model.py --output-dir vlm_hf_model
optimum-cli export openvino --model vlm_hf_model --task image-text-to-text \
    --weight-format fp32 vlm_ov_model
convert_tokenizer --with-detokenizer -o vlm_ov_model vlm_hf_model
```

---

## Using the models

### LLM with openvino_genai

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("ov_model", "CPU")
print(pipe.generate("Hello", max_new_tokens=50, do_sample=False))
# → For the night is dark and full of terrors...For the night is dark …

sched = ov_genai.SchedulerConfig()
sched.max_num_batched_tokens = 256
sched.cache_size = 1
cb = ov_genai.ContinuousBatchingPipeline("ov_model", sched, "CPU")
print(cb.generate(["Hello"], [ov_genai.GenerationConfig(max_new_tokens=50)])[0].m_generation_ids[0])
```

### VLM with openvino_genai

```python
import numpy as np
import openvino as ov
import openvino_genai as ov_genai

pipe = ov_genai.VLMPipeline("vlm_ov_model", "CPU")
# images must be a list of ov.Tensor (HxWxC uint8)
img = ov.Tensor(np.zeros((224, 224, 3), dtype=np.uint8))
print(pipe.generate("What do you see?", images=[img], max_new_tokens=40, do_sample=False))
# → For the night is dark and full of terrors...
```

### With OVMS export_model.py

```bash
# Either a local build (ov_model/) or the published HF repo works directly.
# Point your OVMS config at it with pipeline_type LM or LM_CB.
python3 demos/common/export_models/export_model.py text_generation \
    --source_model mzeglars/dummy-cyclic-gpt2-ov \
    --model_repository_path models \
    --pipeline_type LM_CB
```

---

## Migration status in OVMS tests

`src/test/llm/llmnode_test.cpp` and its VLM counterpart
(`src/test/llm/visual_language_model/initialization_test.cpp`) have been
migrated to use these dummy models instead of downloading real ones
(`HuggingFaceTB/SmolLM2-360M-Instruct`, `facebook/opt-125m`,
`OpenVINO/InternVL2-1B-int4-ov`). `OpenVINO/InternVL2-1B-int4-ov` download has
been removed entirely from `prepare_llm_models.sh` /
`windows_prepare_llm_models.bat` since nothing references it anymore; instead,
both scripts now pull `mzeglars/dummy-cyclic-gpt2-ov` and
`mzeglars/dummy-cyclic-llava-ov` from HF into
`src/test/llm_testing/mzeglars/...` (skipped if already present, like every
other model in that script). Tests reference the models at
`/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-gpt2-ov` and
`/ovms/src/test/llm_testing/mzeglars/dummy-cyclic-llava-ov`.

**Left on real models (not migrated), and why:**
- `LLMStartWithTaskParameter` tests (`llmnode_test.cpp`) still use
  `SmolLM2-360M-Instruct` — OVMS's `--task text_generation` auto-detection
  keys off architectures ending in `ForCausalLM`/`ForConditionalGeneration`
  (see `TextGenerationDetector::scan` in `src/default_task_detector.cpp`), and
  the dummy LLM is `GPT2LMHeadModel`, which isn't recognized. Fixing this would
  require either changing the dummy model's reported architecture or adding it
  to the detector's allow-list.
- `lm_cb_with_tool_parser.pbtxt` / tool-call parsing tests still use
  `facebook/opt-125m` — tool-call tests assert on real generated JSON content;
  while guided/grammar-constrained decoding should make this model-agnostic in
  theory, it hasn't been verified against the 1-layer dummy model yet.
- `lm_cb_speculative.pbtxt`, `lm_cb_prompt_lookup.pbtxt` and
  `assisted_decoding_test.cpp` still use `facebook/opt-125m` for both main and
  draft model — speculative/assisted decoding tests compare token-level
  behavior between runs and haven't been validated against the dummy model.
- `facebook/opt-125m` and `HuggingFaceTB/SmolLM2-360M-Instruct` remain in
  `prepare_llm_models.sh` because other test files still depend on them for
  properties the dummy model doesn't have, e.g. `opt-125m` having **no** chat
  template (`chat_template_processor_test.cpp`, `input_processing_integration_test.cpp`,
  `chat_template_end_to_end_jinja_test.cpp`, `chat_template_end_to_end_minja_test.cpp`,
  `chat_template_and_parser_onyx_roundtrip_test.cpp`) and as a generic tokenizer
  fixture in several `output_parsers/*_test.cpp` files, `text_streamer_test.cpp`,
  `http_openai_handler_test.cpp`, `llmtemplate_test.cpp`, `graph_export_test.cpp`.
  Migrating those would require a second dummy variant without a chat template.
