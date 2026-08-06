# Qwen3-Omni-30B-A3B-Instruct Export Reproduction Steps

Tested on: 2026-08-06, Ubuntu with Python 3.12

## Summary

Export of `Qwen/Qwen3-Omni-30B-A3B-Instruct` to OpenVINO IR (int4 weights, u8 KV cache)
using `demos/common/export_models/export_model.py`.

**Requirements:**
- ~22GB RAM minimum for inference
- ~70.5GB disk space for downloading the source model
- ~150GB total disk space (source + converted model)

## Environment Setup

```bash
cd demos/common/export_models
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Package Versions

The following package versions were verified to work:

| Package | Version | Notes |
|---------|---------|-------|
| transformers | 5.0.0 | **Requires patch** (see below) |
| optimum-intel | 2.1.0 | Must be >=2.0 for qwen3_omni_moe support |
| optimum | 2.3.0 | Installed as dependency of optimum-intel 2.1.0 |
| openvino | 2026.3.0 | |
| openvino-tokenizers | 2026.3.0.0 | |
| nncf | 3.3.0 | |
| torch | 2.11.0+cpu | |
| safetensors | 0.7.0 | Required by optimum-intel 2.1.0 (<0.8.0) |

### Upgrading optimum-intel

The default `requirements.txt` installs `optimum-intel==1.27.0.dev0` which does **not** support
`qwen3_omni_moe` architecture. You must upgrade:

```bash
pip install "optimum-intel[openvino]>=2.0"
```

This will install `optimum-intel==2.1.0`, `optimum==2.3.0`, and downgrade `safetensors` to 0.7.0.

## Required Patch: transformers 5.0.0

`transformers==5.0.0` has a bug in `Qwen3OmniMoeTalkerCodePredictorConfig.__init__` where
line 592 references `self.use_sliding_window` which doesn't exist in that class:

```python
# Bug on line 592 of:
# .venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py
self.sliding_window = sliding_window if self.use_sliding_window else None
#                                       ^^^^^^^^^^^^^^^^^^^^^^^ AttributeError
```

Line 586 already correctly sets `self.sliding_window = sliding_window`, so line 592 is a
duplicate that should be removed.

**Apply the patch:**

```bash
sed -i 's/self.sliding_window = sliding_window if self.use_sliding_window else None/# PATCHED: removed buggy use_sliding_window reference/' \
  .venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py
```

## Model Selection

- **`Qwen/Qwen3-Omni-30B-A3B-Instruct`** — Works. Full model with thinker + talker components.
- **`Qwen/Qwen3-Omni-30B-A3B-Thinking`** — Does NOT work. Missing `talker` submodel, causes
  `AttributeError: 'Qwen3OmniMoeForConditionalGeneration' object has no attribute 'talker'`
  in optimum-intel's export pipeline.

## Export Command

```bash
source .venv/bin/activate

python export_model.py text_generation \
    --source_model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --weight-format int4 \
    --kv_cache_precision u8 \
    --config_file_path models/config_all.json \
    --model_repository_path models
```

**Important:** The venv must be activated (not just using `.venv/bin/python`) because
`export_model.py` shells out to `optimum-cli` as a subprocess, and it needs to find the
venv's `optimum-cli` on `$PATH` — otherwise it picks up the system-level one which may
have an incompatible transformers version.

## Expected Output

The export produces the following files in `models/Qwen/Qwen3-Omni-30B-A3B-Instruct/`:

```
chat_template.jinja
config.json
generation_config.json
graph.pbtxt
openvino_audio_encoder_model.bin
openvino_audio_encoder_model.xml
openvino_code2wav_model.bin
openvino_code2wav_model.xml
openvino_code_predictor_model.bin
openvino_code_predictor_model.xml
openvino_config.json
openvino_detokenizer.bin
openvino_detokenizer.xml
openvino_language_model.bin
openvino_language_model.xml
openvino_talker_model.bin
openvino_talker_model.xml
openvino_talker_projections_model.bin
openvino_talker_projections_model.xml
openvino_talker_text_embeddings_model.bin
openvino_talker_text_embeddings_model.xml
openvino_text_embeddings_model.bin
openvino_text_embeddings_model.xml
openvino_tokenizer.bin
openvino_tokenizer.xml
openvino_vision_embeddings_model.bin
openvino_vision_embeddings_model.xml
openvino_vision_embeddings_pos_model.bin
openvino_vision_embeddings_pos_model.xml
preprocessor_config.json
processor_config.json
tokenizer_config.json
tokenizer.json
```

Weight compression summary:
- Language model: int4_asym (group size 128) — 384 layers + 1 layer int8_asym
- Talker model: int8_sym — 5301 layers
- Audio encoder: int8_sym — 117 layers
- Code predictor: int8_sym — 198 layers
- Other submodels: int8_sym

## Running the Server

```bash
# Docker
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v $(pwd)/models:/models:rw \
  openvino/model_server:weekly \
  --rest_port 8000 --config_path /models/config_all.json

# Bare metal
ovms --rest_port 8000 --config_path models/config_all.json
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'Qwen3OmniMoeTalkerCodePredictorConfig' object has no attribute 'use_sliding_window'` | Bug in transformers 5.0.0 | Apply the sed patch above |
| `ValueError: Trying to export a qwen3_omni_moe model, that is a custom or unsupported architecture` | optimum-intel too old | Upgrade to `optimum-intel>=2.0` |
| `AttributeError: 'Qwen3OmniMoeForConditionalGeneration' object has no attribute 'talker'` | Using the "Thinking" variant | Use `Qwen3-Omni-30B-A3B-Instruct` instead |
| `KeyError: 'qwen3_omni_moe'` from system-level transformers | System `optimum-cli` used instead of venv | Activate venv with `source .venv/bin/activate` before running |
