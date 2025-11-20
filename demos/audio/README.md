# How to serve audio models via OpenAI API {#ovms_demos_audio}

This demo shows how to deploy audio models in the OpenVINO Model Server.
Speech generation and speech recognition models are exposed via OpenAI API `audio/speech`, `audio/transcriptions` and `audio/translations` endpoints.

Check supported [Speech Recognition Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) and [Speech Generation Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models).

## Prerequisites

**OVMS version 2025.4** This demo require version 2025.4 or nightly release.

**Model preparation**: Python 3.10 or higher with pip

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

**Client**: curl or Python for using OpenAI client package

## Speech generation
### Model preparation
Supported models should use the topology of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) which needs to be converted to IR format before using in OVMS.

Specific OVMS pull mode example for models requiring conversion is described in the [Ovms pull mode](../../docs/pull_hf_models.md#pulling-models-outside-openvino-organization)

Or you can use the python export_model.py script described below.

Here, the original Text to Speech model will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
Execution parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py text2speech --source_model microsoft/speecht5_tts --weight-format fp32 --model_name microsoft/speecht5_tts --config_file_path models/config.json --model_repository_path models --overwrite_models --vocoder microsoft/speecht5_hifigan
```

> **Note:** Change the `--weight-format` to quantize the model to `fp16` or `int8` precision to reduce memory consumption and improve performance.

### Deployment

**CPU**

Running this command starts the container with CPU only target device:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --model_path /models/microsoft/speecht5_tts --model_name microsoft/speecht5_tts
```

**Deploying on Bare Metal**

```bat
mkdir models
ovms --rest_port 8000 --source_model microsoft/speecht5_tts --model_repository_path models --model_name microsoft/speecht5_tts --task text2speech --target_device CPU
```

### Request Generation 

:::{dropdown} **Unary call with curl**


```bash
curl http://localhost:8125/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish\", \"input\": \"¿Qué es OpenVino?\"}" -o speech_spanish.wav
```
:::

:::{dropdown} **Unary call with OpenAi python library**

```python
from pathlib import Path
from openai import OpenAI

prompt = "The quick brown fox jumped over the lazy dog"
filename = "speech.wav"
url="http://localhost:8000/v3"


speech_file_path = Path(__file__).parent / "speech.wav"
client = OpenAI(base_url=url, api_key="not_used")

with client.audio.speech.with_streaming_response.create(
  model="microsoft/speecht5_tts",
  voice="unused",
  input=prompt
) as response:
  response.stream_to_file(speech_file_path)


print("Generation finished")
```
:::

Play speech.wav file to check generated speech.

## Transcription
### Model preparation
Many Whisper models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/collections/OpenVINO/speech-to-text) and used both for translations and transcriptions endpoints.
But in this demo we will use openai/whisper-large-v3-turbo which needs to be converted to IR format before using in OVMS.
Specific OVMS pull mode example for models requiring conversion is described in the [Ovms pull mode](../../docs/pull_hf_models.md#pulling-models-outside-openvino-organization)

Or you can use the python export_model.py script described below.

Here, the original Speech to Text model will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
Execution parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py speech2text --source_model openai/whisper-large-v3-turbo --weight-format fp32 --model_name openai/whisper-large-v3-turbo --config_file_path models/config.json --model_repository_path models --overwrite_models
```

> **Note:** Change the `--weight-format` to quantize the model to `fp16` or `int8` precision to reduce memory consumption and improve performance.

### Deployment

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
mkdir -p modelsspeech2text
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --source_model openai/whisper-large-v3-turbo --model_repository_path /models --model_name openai/whisper-large-v3-turbo --task 
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support.
It can be applied using the commands below:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --source_model openai/whisper-large-v3-turbo --model_repository_path models --model_name openai/whisper-large-v3-turbo --task speech2text --target_device GPU
```
:::

:::{dropdown} **Deploying on Bare Metal**

If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
mkdir models
ovms --rest_port 8000 --source_model openai/whisper-large-v3-turbo --model_repository_path models --model_name openai/whisper-large-v3-turbo --task speech2text --target_device CPU
```
or
```bat
ovms --rest_port 8000 --source_model openai/whisper-large-v3-turbo --model_repository_path models --model_name openai/whisper-large-v3-turbo --task speech2text --target_device GPU
```
:::

### Request Generation 
Transcript file that was previously generated with audio/speech endpoint.

:::{dropdown} **Unary call with curl**


```bash
curl http://localhost:8000/v3/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@speech.wav" -F model="OpenVINO/whisper-base-fp16-ov"
```
```json
{"text": " The quick brown fox jumped over the lazy dog."}
```
:::

:::{dropdown} **Unary call with python OpenAI library**

```python
from pathlib import Path
from openai import OpenAI

filename = "speech.wav"
url="http://localhost:8000/v3"


speech_file_path = Path(__file__).parent / filename
client = OpenAI(base_url=url, api_key="not_used")

audio_file = open(filename, "rb")
transcript = client.audio.transcriptions.create(
  model="OpenVINO/whisper-base-fp16-ov",
  file=audio_file
)

print(transcript.text)
```
```
The quick brown fox jumped over the lazy dog.
```
:::

## Translation
To test translations endpoint we first need to prepare audio file with speech in language other than english, e.g. spanish. To generate such sample we will use finetuned version of microsoft/speecht5_tts model.

```console
python export_model.py text2speech --source_model Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish --weight-format fp32 --model_name speecht5_tts_spanish --config_file_path models/config.json --model_repository_path models --overwrite_models --vocoder microsoft/speecht5_hifigan
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --model_path /models/Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish --model_name speecht5_tts_spanish
curl http://localhost:8125/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"speecht5_tts_spanish\", \"input\": \"Madrid es la capital de España\"}" -o speech_spanish.wav
```

### Model preparation
Whisper models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/collections/OpenVINO/speech-to-text) and used both for translations and transcriptions endpoints.
Here is an example of OpenVINO/whisper-base-fp16-ov deployment:

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8125:8125 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8125 --source_model OpenVINO/whisper-base-fp16-ov --model_repository_path /models --model_name OpenVINO/whisper-base-fp16-ov --task speech2text
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support.
It can be applied using the commands below:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --source_model OpenVINO/whisper-base-fp16-ov --model_repository_path models --model_name OpenVINO/whisper-base-fp16-ov --task speech2text --target_device GPU
```
:::

:::{dropdown} **Deploying on Bare Metal**

If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
mkdir models
ovms --rest_port 8000 --source_model OpenVINO/whisper-base-fp16-ov --model_repository_path models --model_name OpenVINO/whisper-base-fp16-ov --task speech2text --target_device CPU
```
or
```bat
ovms --rest_port 8000 --source_model OpenVINO/whisper-base-fp16-ov --model_repository_path models --model_name OpenVINO/whisper-base-fp16-ov --task speech2text --target_device GPU
```
:::

### Request Generation 
Translate file that was previously generated with audio/speech endpoint.

:::{dropdown} **Unary call with curl**


```bash
curl http://localhost:8125/v3/audio/translations -H "Content-Type: multipart/form-data" -F file="@speech_spanish.wav" -F model="OpenVINO/whisper-base-fp16-ov"
```
```json
{"text": " Madrid is the capital of Spain."}
```
:::

:::{dropdown} **Unary call with python OpenAI library**

```python
from pathlib import Path
from openai import OpenAI

filename = "speech_spanish.wav"
url="http://localhost:8000/v3"


speech_file_path = Path(__file__).parent / filename
client = OpenAI(base_url=url, api_key="not_used")

audio_file = open(filename, "rb")
translation = client.audio.translations.create(
  model="OpenVINO/whisper-base-fp16-ov",
  file=audio_file
)

print(translation.text)
```
```
Madrid is the capital of Spain.
```
:::