# Omni Pipeline Demo

Example clients for Qwen3-Omni models served by OpenVINO Model Server. The native Omni pipeline accepts text, images, and audio and can return both text and generated speech.

This demo uses the preconverted [OpenVINO/Qwen3-Omni-30B-A3B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen3-Omni-30B-A3B-Instruct-int4-ov) model. Hugging Face access is required while the model repository is private.

## Prerequisites

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [bare metal deployment guide](../../docs/deploying_server_baremetal.md).

**Client**: Install the Python dependencies:

```console
pip install openai numpy sounddevice
```

## Server Deployment

OVMS auto-detects the native, stateful `OMNI` pipeline from the model's talker assets.

:::{dropdown} **Deploying with Docker**

```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:rw \
  openvino/model_server:weekly --rest_port 8000 \
  --source_model OpenVINO/Qwen3-Omni-30B-A3B-Instruct-int4-ov \
  --model_repository_path /models \
  --task text_generation
```

:::

:::{dropdown} **Deploying on Bare Metal**

```text
ovms --rest_port 8000 \
  --source_model OpenVINO/Qwen3-Omni-30B-A3B-Instruct-int4-ov \
  --model_repository_path <models-directory> \
  --task text_generation
```

:::

Omni models can also be explicitly loaded as `VLM_CB` for continuous batching with text, image, and audio input, but that mode produces text only.

See the [Omni model reference](../../docs/omni.md) for the model layout, request formats, and API details.

## Readiness Check

Wait for the model to load, then verify the model list:

```console
curl http://localhost:8000/v1/models
```

## Request Generation

### Chat Completions API

```text
# Text → text
python3 chat_completions.py --prompt "What is OpenVINO?"

# Text → text + audio
python3 chat_completions.py --prompt "Say 3 sentences about France. Dont use smileys, pure text." --audio-output --voice m36 --save output.wav

# Text + audio → text
python3 chat_completions.py --audio recording.wav --prompt "what is in the recording?"

# Audio + image + text → text + audio
python3 chat_completions.py --audio recording.wav --image ./../common/static/images/gorilla.jpeg --prompt "Detect which animal is in the image and tell me product name which is described in the audio." --audio-output --voice m02 --save output.wav
```

### Responses API

```text
# Text → text
python3 responses.py --prompt "What is OpenVINO?"

# Text → text + audio (streaming with playback)
python3 responses.py --prompt "Say 3 sentences about France. Dont use smileys, pure text." --audio-output --voice m31 --save output.wav

# Audio → text (unary)
python3 responses.py --audio recording.wav --prompt "what is in the recording?"

# Image + text → text + audio, streaming
python3 responses.py --image ./../common/static/images/gorilla.jpeg --prompt "explain to me what is in the image" --audio-output --stream --voice m02 --save output.wav

# Audio + text → text + audio, streaming
python3 responses.py --audio recording.wav --prompt "what is in the recording?" --audio-output --stream --voice m02 --save output.wav

```

## Available Voices

`f04`, `f245`, `f37`, `m02`, `m31`, `m36`, `br_f019` (model-dependent).

## API Reference

[Responses Streaming API](https://developers.openai.com/api/reference/resources/responses/streaming-events#response.audio.delta)
