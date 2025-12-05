# How to serve audio models via OpenAI API {#ovms_demos_audio}

This demo shows how to deploy audio models in the OpenVINO Model Server.
Speech generation and speech recognition models are exposed via OpenAI API `audio/speech`, `audio/transcriptions` and `audio/translations` endpoints.

Check supported [Speech Recognition Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) and [Speech Generation Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models).

## Prerequisites

**OVMS version 2025.4** This demo require version 2025.4 or nightly release.

**Model preparation**: Python 3.10 or higher with pip

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

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
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py text2speech --source_model microsoft/speecht5_tts --weight-format fp16 --model_name microsoft/speecht5_tts --config_file_path models/config.json --model_repository_path models --overwrite_models --vocoder microsoft/speecht5_hifigan
```

> **Note:** Change the `--weight-format` to quantize the model to `int8` precision to reduce memory consumption and improve performance.

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [T2s calculator documentation](../../docs/speech_generation/reference.md) to learn more about configuration options and limitations.

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
curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"microsoft/speecht5_tts\", \"input\": \"The quick brown fox jumped over the lazy dog\"}" -o speech.wav
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

## Benchmarking speech generation
An asynchronous benchmarking client can be used to access the model server performance with various load conditions. Below are execution examples captured on Intel(R) Core(TM) Ultra 7 258V.

```console
git clone https://github.com/openvinotoolkit/model_server
cd model_server/demos/benchmark/v3/
pip install -r requirements.txt
python benchmark.py --api_url http://localhost:8122/v3/audio/speech --model microsoft/speecht5_tts --batch_size 1 --limit 100 --request_rate inf --backend text2speech --dataset edinburghcstr/ami --hf-subset 'ihm' --tokenizer openai/whisper-large-v3-turbo --trust-remote-code True
Number of documents: 100
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [01:58<00:00,  1.19s/it]
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Tokens: 1802
Success rate: 100.0%. (100/100)
Throughput - Tokens per second: 15.2
Mean latency: 63653.98 ms
Median latency: 66736.83 ms
Average document length: 18.02 tokens
```

## Transcription
### Model preparation
Many variances of Whisper models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/collections/OpenVINO/speech-to-text) and used both for translations and transcriptions endpoints.
However in this demo we will use openai/whisper-large-v3-turbo which needs to be converted to IR format before using in OVMS.

Here, the original Speech to Text model will be converted to IR format and quantized.
That ensures faster initialization time, better performance and lower memory consumption.
Execution parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py speech2text --source_model openai/whisper-large-v3-turbo --weight-format fp16 --model_name openai/whisper-large-v3-turbo --config_file_path models/config.json --model_repository_path models --overwrite_models
```

> **Note:** Change the `--weight-format` to quantize the model to `int8` precision to reduce memory consumption and improve performance.

### Deployment

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --source_model openai/whisper-large-v3-turbo --model_repository_path /models --model_name openai/whisper-large-v3-turbo --task speech2text
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

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [S2t calculator documentation](../../docs/speech_recognition/reference.md) to learn more about configuration options and limitations.

### Request Generation 
Transcript file that was previously generated with audio/speech endpoint.

:::{dropdown} **Unary call with curl**


```bash
curl http://localhost:8000/v3/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@speech.wav" -F model="openai/whisper-large-v3-turbo"
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
  model="openai/whisper-large-v3-turbo",
  file=audio_file
)

print(transcript.text)
```
```
The quick brown fox jumped over the lazy dog.
```
:::

## Benchmarking transcription
An asynchronous benchmarking client can be used to access the model server performance with various load conditions. Below are execution examples captured on Intel(R) Core(TM) Ultra 7 258V.

```console
git clone https://github.com/openvinotoolkit/model_server
cd model_server/demos/benchmark/v3/
pip install -r requirements.txt
python benchmark.py --api_url http://localhost:8000/v3/audio/transcriptions --model openai/whisper-large-v3-turbo --batch_size 1 --limit 1000 --request_rate inf --dataset edinburghcstr/ami --hf-subset ihm --backend speech2text --trust-remote-code True
Number of documents: 1000
100%|██████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:44<00:00,  3.51it/s]
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Tokens: 10948
Success rate: 100.0%. (1000/1000)
Throughput - Tokens per second: 38.5
Mean latency: 26670.64 ms
Median latency: 20772.09 ms
Average document length: 10.948 tokens
```

## Translation
To test translations endpoint we first need to prepare audio file with speech in language other than English, e.g. Spanish. To generate such sample we will use finetuned version of microsoft/speecht5_tts model.

```console
python export_model.py text2speech --source_model Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish --weight-format fp16 --model_name speecht5_tts_spanish --config_file_path models/config.json --model_repository_path models --overwrite_models --vocoder microsoft/speecht5_hifigan

docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --model_path /models/Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish --model_name speecht5_tts_spanish

curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"speecht5_tts_spanish\", \"input\": \"Madrid es la capital de España\"}" -o speech_spanish.wav
```

### Model preparation
Whisper models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/collections/OpenVINO/speech-to-text) and used both for translations and transcriptions endpoints.
Here is an example of OpenVINO/whisper-large-v3-fp16-ov deployment:

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --source_model OpenVINO/whisper-large-v3-fp16-ov --model_repository_path /models --model_name OpenVINO/whisper-large-v3-fp16-ov --task speech2text
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support.
It can be applied using the commands below:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --source_model OpenVINO/whisper-large-v3-fp16-ov --model_repository_path models --model_name OpenVINO/whisper-large-v3-fp16-ov --task speech2text --target_device GPU
```
:::

:::{dropdown} **Deploying on Bare Metal**

If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
mkdir models
ovms --rest_port 8000 --source_model OpenVINO/whisper-large-v3-fp16-ov --model_repository_path models --model_name OpenVINO/whisper-large-v3-fp16-ov --task speech2text --target_device CPU
```
or
```bat
ovms --rest_port 8000 --source_model OpenVINO/whisper-large-v3-fp16-ov --model_repository_path models --model_name OpenVINO/whisper-large-v3-fp16-ov --task speech2text --target_device GPU
```
:::

### Request Generation 
Transcript and translate file that was previously generated with audio/speech endpoint.

:::{dropdown} **Unary call with curl**


```bash
curl http://localhost:8000/v3/audio/translations -H "Content-Type: multipart/form-data" -F file="@speech_spanish.wav" -F model="OpenVINO/whisper-large-v3-fp16-ov"
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
  model="OpenVINO/whisper-large-v3-fp16-ov",
  file=audio_file
)

print(translation.text)
```
```
Madrid is the capital of Spain.
```
:::