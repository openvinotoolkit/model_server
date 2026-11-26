# How to serve audio models via OpenAI API {#ovms_demos_audio}

This demo shows how to deploy audio models in the OpenVINO Model Server.
Speech generation and speech recognition models are exposed via OpenAI API `audio/speech`, `audio/transcriptions` and `audio/translations` endpoints.
Speech-to-text streaming responses are supported for `audio/transcriptions` endpoint.

Check supported [Speech Recognition Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) and [Speech Generation Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models).

## Prerequisites

**OVMS version 2026.3** This demo requires version 2026.3 or nightly release.

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**Client**: curl or Python for using OpenAI client package

Install the client dependency before running the Python examples:

```console
pip install openai
```

## Speech generation

Supported models and variants:

- **Kokoro** - primary example in this demo; uses predefined speaker embeddings from the model's `voices/` directory.
- **SpeechT5** - supported for existing deployments.
- **Qwen3 TTS CustomVoice** - selects a model-defined speaker with `voice` and optionally accepts `instructions` to guide delivery.
- **Qwen3 TTS VoiceDesign** - creates a voice from `instructions`; no `voice` is required.
- **Qwen3 TTS Base** - not supported because reference-audio inputs are outside the OpenAI-compatible `audio/speech` request schema.

For Kokoro requests, `language` selects the phonemization path. Supported values are `en-us`, `en-gb`, `es`, `fr-fr`, `it`, `pt-br`, and `hi`; the default is `en-us`.

### Model preparation
This demo uses the official OpenVINO model [OpenVINO/Kokoro-82M-int8-ov](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov) available on HuggingFace.
The model can be pulled directly by OVMS without any conversion step.

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before starting OVMS to connect to the HF Hub.
> **Note:** Kokoro voices are loaded from the `voices/` directory in the model repository. OVMS loads each voice file under the filename without the extension, for example `af_alloy` for `af_alloy.bin`.

See the [T2s calculator documentation](../../docs/speech_generation/reference.md) to learn more about configuration options and limitations.

### Deployment

Select the model to deploy. Replace `<ovms_model>` with the OpenVINO model repository or local model path when the converted model is not published on Hugging Face.

::::{tab-set}
:::{tab-item} Kokoro
:sync: Kokoro
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:latest \
  --rest_port 8000 \
  --source_model OpenVINO/Kokoro-82M-int8-ov \
  --model_repository_path /models \
  --model_name Kokoro-82M-int8-ov \
  --target_device CPU
```
:::
:::{tab-item} SpeechT5
:sync: SpeechT5
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:latest \
  --rest_port 8000 \
  --source_model <ovms_model> \
  --model_repository_path /models \
  --model_name speecht5 \
  --task text2speech
```
:::
:::{tab-item} Qwen3 TTS CustomVoice
:sync: Qwen3 TTS CustomVoice
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:latest \
  --rest_port 8000 \
  --source_model <ovms_model> \
  --model_repository_path /models \
  --model_name qwen3-tts-customvoice \
  --task text2speech
```
:::
:::{tab-item} Qwen3 TTS VoiceDesign
:sync: Qwen3 TTS VoiceDesign
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:latest \
  --rest_port 8000 \
  --source_model <ovms_model> \
  --model_repository_path /models \
  --model_name qwen3-tts-voicedesign \
  --task text2speech
```
:::
::::

For Kokoro on NPU, add `--device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` to the Docker command, use the `openvino/model_server:latest-gpu` image, and set `--target_device NPU`. On bare metal, use the same selected model arguments with local repository paths.

### Request Generation

Select the model to show the matching request examples:

::::{tab-set}
:::{tab-item} Kokoro
:sync: Kokoro
**cURL**
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kokoro-82M-int8-ov",
    "voice": "af_alloy",
    "input": "The quick brown fox jumped over the lazy dog."
  }' \
  -o speech.wav
```

**OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with client.audio.speech.with_streaming_response.create(
    model="Kokoro-82M-int8-ov",
    voice="af_alloy",
    input="The quick brown fox jumped over the lazy dog.",
) as response:
    response.stream_to_file("speech.wav")
```
:::
:::{tab-item} SpeechT5
:sync: SpeechT5
**cURL**
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speecht5",
    "voice": "default",
    "input": "The quick brown fox jumped over the lazy dog."
  }' \
  -o speech.wav
```

**OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with client.audio.speech.with_streaming_response.create(
    model="speecht5",
    voice="default",
    input="The quick brown fox jumped over the lazy dog.",
) as response:
    response.stream_to_file("speech.wav")
```
:::
:::{tab-item} Qwen3 TTS CustomVoice
:sync: Qwen3 TTS CustomVoice
**cURL**
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-customvoice",
    "voice": "<model-speaker>",
    "instructions": "Speak warmly and at a measured pace.",
    "input": "Welcome to OpenVINO Model Server."
  }' \
  -o custom_voice.wav
```

**OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts-customvoice",
    voice="<model-speaker>",
    instructions="Speak warmly and at a measured pace.",
    input="Welcome to OpenVINO Model Server.",
) as response:
    response.stream_to_file("custom_voice.wav")
```
:::
:::{tab-item} Qwen3 TTS VoiceDesign
:sync: Qwen3 TTS VoiceDesign
**cURL**
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-voicedesign",
    "instructions": "A calm, clear adult voice with a warm conversational tone.",
    "input": "Welcome to OpenVINO Model Server."
  }' \
  -o designed_voice.wav
```

**OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts-voicedesign",
    instructions="A calm, clear adult voice with a warm conversational tone.",
    input="Welcome to OpenVINO Model Server.",
) as response:
    response.stream_to_file("designed_voice.wav")
```
:::
::::

Play the generated audio file to verify the response.

## Benchmarking speech generation
For synthetic and real-text benchmark examples, setup, and RTFx interpretation, see
[Speech Generation](../benchmark/openai/README.md#speech-generation) in the benchmark guide.

## Transcription

### Model selection and deployment

Preconfigured speech-to-text models can be pulled directly from the [OpenVINO Hugging Face organization](https://huggingface.co/collections/OpenVINO/speech-to-text). Select a model below and run the matching command. Each model name is also used in transcription requests.

::::{tab-set}
:::{tab-item} Whisper Large v3
:sync: whisper-large-v3-int8-ov
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:weekly \
  --rest_port 8000 \
  --source_model OpenVINO/whisper-large-v3-int8-ov \
  --model_name whisper-large-v3-int8-ov \
  --model_repository_path /models
```
:::
:::{tab-item} Whisper Large v3 Turbo
:sync: whisper-large-v3-turbo-int8-ov
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:weekly \
  --rest_port 8000 \
  --source_model OpenVINO/whisper-large-v3-turbo-int8-ov \
  --model_name whisper-large-v3-turbo-int8-ov \
  --model_repository_path /models
```
:::
:::{tab-item} Distil-Whisper
:sync: distil-whisper-large-v3-int8-ov
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:weekly \
  --rest_port 8000 \
  --source_model OpenVINO/distil-whisper-large-v3-int8-ov \
  --model_name distil-whisper-large-v3-int8-ov \
  --model_repository_path /models
```
:::
:::{tab-item} Qwen3 ASR
:sync: qwen3-asr
```bash
mkdir -p ${HOME}/models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 \
  -v ${HOME}/models:/models:rw openvino/model_server:weekly \
  --rest_port 8000 \
  --source_model <ovms_model> \
  --model_name qwen3-asr \
  --model_repository_path /models \
  --task text_generation
```

OVMS automatically selects `VLM_CB` for Qwen3 ASR on CPU and GPU. Add `--pipeline_type VLM_CB` to make that selection explicit.
:::
::::

The same model selection can be used with the bare-metal binary by replacing the container and mount arguments with local paths. To use GPU in Docker, add `--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)` to the command.

### Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/models
```

The default configuration should work in most cases but the parameters can be tuned via OVMS arguments. See the [s2t calculator documentation](../../docs/speech_recognition/reference.md) to learn more about configuration options and limitations.

### Request Generation 
Transcribe the speech.wav file generated in the [Speech generation](#speech-generation) section.

> **Note:** Streaming responses are supported for `audio/transcriptions`. `audio/translations` does not support streaming.

::::{tab-set}
:::{tab-item} Whisper Large v3
:sync: whisper-large-v3-int8-ov
**Unary call with cURL**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=whisper-large-v3-int8-ov' \
  -F 'language=en'
```

**Unary call with OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with open("speech.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3-int8-ov",
        language="en",
        file=audio_file,
    )
print(transcript.text)
```

**Streaming call with cURL**
```bash
curl -N http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=whisper-large-v3-int8-ov' \
  -F 'language=en' \
  -F 'stream=true'
```

**Sentence timestamps**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=whisper-large-v3-int8-ov' \
  -F 'language=en' \
  -F 'timestamp_granularities[]=segment'
```
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with open("speech.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3-int8-ov",
        language="en",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
        file=audio_file,
    )
print(transcript.segments)
```
:::
:::{tab-item} Whisper Large v3 Turbo
:sync: whisper-large-v3-turbo-int8-ov
**Unary call with cURL**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=whisper-large-v3-turbo-int8-ov' \
  -F 'language=en'
```

**Unary call with OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with open("speech.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo-int8-ov",
        language="en",
        file=audio_file,
    )
print(transcript.text)
```

**Streaming call with cURL**
```bash
curl -N http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=whisper-large-v3-turbo-int8-ov' \
  -F 'language=en' \
  -F 'stream=true'
```

**Sentence timestamps**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=whisper-large-v3-turbo-int8-ov' \
  -F 'language=en' \
  -F 'timestamp_granularities[]=segment'
```
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with open("speech.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo-int8-ov",
        language="en",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
        file=audio_file,
    )
print(transcript.segments)
```
:::
:::{tab-item} Distil-Whisper
:sync: distil-whisper-large-v3-int8-ov
**Unary call with cURL**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=distil-whisper-large-v3-int8-ov' \
  -F 'language=en'
```

**Unary call with OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with open("speech.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="distil-whisper-large-v3-int8-ov",
        language="en",
        file=audio_file,
    )
print(transcript.text)
```

**Streaming call with cURL**
```bash
curl -N http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=distil-whisper-large-v3-int8-ov' \
  -F 'language=en' \
  -F 'stream=true'
```

**Sentence timestamps**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=distil-whisper-large-v3-int8-ov' \
  -F 'language=en' \
  -F 'timestamp_granularities[]=segment'
```
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with open("speech.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="distil-whisper-large-v3-int8-ov",
        language="en",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
        file=audio_file,
    )
print(transcript.segments)
```
:::
:::{tab-item} Qwen3 ASR
:sync: qwen3-asr
**Unary call with cURL**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=qwen3-asr'
```

**Unary call with OpenAI Python client**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not_used")
with open("speech.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="qwen3-asr",
        file=audio_file,
    )
print(transcript.text)
```

**Streaming call with cURL**
```bash
curl -N http://localhost:8000/v1/audio/transcriptions \
  -F 'file=@speech.wav' \
  -F 'model=qwen3-asr' \
  -F 'stream=true'
```

Qwen3 ASR does not expose Whisper's `timestamp_granularities` fields through this endpoint.
:::
::::

Example unary response:
```json
{"text": " The quick brown fox jumped over the lazy dog."}
```

Example streamed chunks:
```text
data: {"type":"transcript.text.delta","delta":"The quick ","logprobs":[]}

data: {"type":"transcript.text.delta","delta":"brown fox ","logprobs":[]}

data: {"type":"transcript.text.done","text":"The quick brown fox jumped over the lazy dog.","logprobs":[]}
```

### Qwen3 ASR continuous batching

`VLM_CB` is the default Qwen3 ASR pipeline on CPU and GPU. Whisper is usually the better choice for a small number of concurrent clients, while Qwen3 ASR continuous batching is intended for higher concurrency.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F 'model=qwen3-asr' \
  -F 'file=@speech.wav'
```

This mode supports transcription but not the `/audio/translations` endpoint.

### Word timestamps
If you need word-level timestamps support, export the model with `export_model.py` and enable this feature during export.

> **Model preparation:** Python 3.10 or higher with pip is required to run `export_model.py`.

Prepare export script and dependencies:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
```

Export Speech-to-Text model with word timestamps enabled.

::::{tab-set}
:::{tab-item} Linux
:sync: Linux
```bash
mkdir -p ${HOME}/models
python export_model.py speech2text --source_model openai/whisper-large-v3-turbo --weight-format fp16 --model_name whisper-large-v3-turbo-word-ts --config_file_path ${HOME}/models/config.json --model_repository_path ${HOME}/models --overwrite_models --enable_word_timestamps
```
:::
:::{tab-item} Windows
:sync: Windows
```bat
mkdir c:\models
python export_model.py speech2text --source_model openai/whisper-large-v3-turbo --weight-format fp16 --model_name whisper-large-v3-turbo-word-ts --config_file_path c:\models\config.json --model_repository_path c:\models --overwrite_models --enable_word_timestamps
```
:::
::::

:::{dropdown} **Deploying with Docker**
```bash
# in case GPU is available
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)

docker run -d ${GPU_ARGS} -u $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly --rest_port 8000 --config_path /models/config.json
```
:::

:::{dropdown} **Deploying on Bare Metal**

The same command can be used for CPU and GPU deployments.

```bat
mkdir c:\models
ovms --rest_port 8000 --config_path c:\models\config.json
```
:::

### Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/models
```

:::{dropdown} **Unary call with cURL (word timestamps)**

```bash
curl http://localhost:8000/v1/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@speech.wav" -F model="whisper-large-v3-turbo-word-ts" -F language="en" -F timestamp_granularities[]="word"
```

Example response:
```json
{"text":" A quick brown fox jumped over the lazy dog","words":[{"word":" A","start":0.0,"end":0.14000000059604645},{"word":" quick","start":0.14000000059604645,"end":0.3400000035762787},{"word":" brown","start":0.3400000035762787,"end":0.7799999713897705},{"word":" fox","start":0.7799999713897705,"end":1.3199999332427979},{"word":" jumped","start":1.3199999332427979,"end":1.7799999713897705},{"word":" over","start":1.7799999713897705,"end":2.0799999237060547},{"word":" the","start":2.0799999237060547,"end":2.259999990463257},{"word":" lazy","start":2.259999990463257,"end":2.5399999618530273},{"word":" dog","start":2.5399999618530273,"end":2.919999837875366}]}
```
:::

:::{dropdown} **Unary call with python OpenAI library (word timestamps)**

```python
from pathlib import Path
from openai import OpenAI

filename = "speech.wav"
url="http://localhost:8000/v1"


speech_file_path = Path(__file__).parent / filename
client = OpenAI(base_url=url, api_key="not_used")

audio_file = open(filename, "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-large-v3-turbo-word-ts",
  language="en",
  response_format="verbose_json",
  timestamp_granularities=["word"],
  file=audio_file
)

print(transcript.text)
print(transcript.words)
```
:::

## Evaluate transcription accuracy and performance with Open ASR Leaderboard

You can evaluate model accuracy (for example WER/CER) against ASR datasets using the Open ASR Leaderboard tooling.

Clone the repository:
```text
git clone https://github.com/huggingface/open_asr_leaderboard.git
cd open_asr_leaderboard
```

Download and apply OVMS API compatibility patch:

    curl -L https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/external/open_asr_leaderboard.patch -o ovms_open_asr_leaderboard.patch
    git apply ovms_open_asr_leaderboard.patch

Set OpenAI-compatible endpoint variables for OVMS:
```text
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY="unused"
```

Install dependencies:
```text
pip install -r requirements/requirements.txt -r requirements/requirements-api.txt openai>=1.0.0 torchcodec==0.12
```

Run evaluation example:
```text
PYTHONPATH=. python api/run_eval.py \
  --model_name openai/whisper-large-v3-turbo-int8-ov \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --max_workers 1 \
  --split test.clean  \
  --dataset "librispeech"
```
Results:
```text
...
Transcribing: 100%|█████████▉| 2617/2620 [12:15<00:00,  5.23it/s]
Transcribing: 100%|█████████▉| 2618/2620 [12:15<00:00,  5.31it/s]
Transcribing: 100%|█████████▉| 2619/2620 [12:16<00:00,  5.41it/s]
Transcribing: 100%|██████████| 2620/2620 [12:16<00:00,  5.20it/s]
Transcribing: 100%|██████████| 2620/2620 [12:16<00:00,  3.56it/s]
Results saved at path: ./results/MODEL_openai-OpenVINO-whisper-large-v3-turbo-int8-ov_DATASET_hf-audio-esb-datasets-test-only-sorted_librispeech_test.clean.jsonl
WER: 1.97 %
RTFx: 28.87
```

Where:
- WER (Word Error Rate) is the percentage of transcription errors compared to the reference text (substitutions + deletions + insertions). Lower is better.
- RTFx (Real-Time Factor, expressed as speedup) indicates processing speed relative to audio duration. Values above 1 mean faster-than-real-time transcription (for example, 5.16 means about 5.16x real time).

**For Open ASR Leaderboard, run `run_eval.py` with model name prefixed by `openai/` (for example `openai/whisper-large-v3-turbo-int8-ov`).**
**OVMS should still be deployed with `--model_name whisper-large-v3-turbo-int8-ov` (evaluation script does not include `openai/` prefix in requests).**
You can replace `librispeech` with other datasets supported by the leaderboard configuration. For multilingual models run_eval_ml.py should be used.

## Translation
To test the translations endpoint you can use your own sample or create one with Kokoro model.
### Sample generation
To prepare an audio file with speech in a language other than English, e.g. Spanish. To generate such a sample, follow the [Speech generation](#speech-generation) section to deploy Kokoro and then run:


**Deploying with Docker**

```bash
mkdir -p ${HOME}/models
# in case GPU is available
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)

docker run -d ${GPU_ARGS} -u $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly --rest_port 8000 --source_model OpenVINO/Kokoro-82M-int8-ov --model_repository_path /models --model_name Kokoro-82M-int8-ov --target_device CPU
```

**Deploying on Bare Metal**

The same command can be used for CPU and GPU deployments. OVMS will auto-detect device settings when `--target_device` is not provided.

```bat
mkdir c:\models
ovms --rest_port 8000 --source_model OpenVINO/Kokoro-82M-int8-ov --model_repository_path c:\models --model_name Kokoro-82M-int8-ov
```

For non-English text, set the `language` field explicitly so OpenVINO GenAI selects the correct phonemization path. Voice selection does not infer the language.

```bash
curl --fail-with-body -sS http://localhost:8000/v1/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"Kokoro-82M-int8-ov\", \"voice\": \"ef_dora\", \"language\": \"es\", \"input\": \"Madrid es la capital de España\"}" -o speech_spanish.wav
```

### Deployment
Whisper models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/collections/OpenVINO/speech-to-text) and used both for translations and transcriptions endpoints.
Here is an example of OpenVINO/whisper-large-v3-int8-ov deployment:

:::{dropdown} **Deploying with Docker**
```bash
mkdir -p ${HOME}/models
# in case GPU is available
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)

docker run -d ${GPU_ARGS} -u $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly --rest_port 8000 --source_model OpenVINO/whisper-large-v3-int8-ov --model_repository_path /models --model_name whisper-large-v3-int8-ov
```
:::

:::{dropdown} **Deploying on Bare Metal**

The same command can be used for CPU and GPU deployments. OVMS will auto-detect device settings when `--target_device` is not provided.

```bat
mkdir c:\models
ovms --rest_port 8000 --source_model OpenVINO/whisper-large-v3-int8-ov --model_repository_path c:\models --model_name whisper-large-v3-int8-ov
```
:::

### Request Generation 
Translate the speech_spanish.wav file generated above.

:::{dropdown} **Unary call with cURL**


```bash
curl http://localhost:8000/v1/audio/translations -H "Content-Type: multipart/form-data" -F file="@speech_spanish.wav" -F model="whisper-large-v3-int8-ov"
```
```json
{"text": " Madrid is the capital of Spain."}
```
:::

:::{dropdown} **Unary call with python OpenAI library**

```code
from pathlib import Path
from openai import OpenAI

filename = "speech_spanish.wav"
url="http://localhost:8000/v1"


speech_file_path = Path(__file__).parent / filename
client = OpenAI(base_url=url, api_key="not_used")

audio_file = open(filename, "rb")
translation = client.audio.translations.create(
  model="whisper-large-v3-int8-ov",
  file=audio_file
)

print(translation.text)
```
```
Madrid is the capital of Spain.
```
:::
