# How to serve audio models via OpenAI API {#ovms_demos_audio}

This demo shows how to deploy audio models in the OpenVINO Model Server.
Speech generation and speech recognition models are exposed via OpenAI API `audio/speech`, `audio/transcriptions` and `audio/translations` endpoints.
Speech-to-text streaming responses are supported for `audio/transcriptions` endpoint.

Check supported [Speech Recognition Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) and [Speech Generation Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models).

## Prerequisites

**OVMS version 2026.3** This demo requires version 2026.3 or nightly release.

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**Client**: curl or Python for using OpenAI client package

## Speech generation
Kokoro is the primary example in this demo, but SpeechT5 remains supported for existing deployments.
### Model preparation
This demo uses a pre-exported OpenVINO IR model [luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS](https://huggingface.co/luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS) available on HuggingFace.
The model can be pulled directly by OVMS without any conversion step.

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before starting OVMS to connect to the HF Hub.
> **Note:** Kokoro voices are loaded from the `voices/` directory in the model repository. OVMS loads each voice file under the filename without the extension, for example `af_alloy` for `af_alloy.bin`.

See the [T2s calculator documentation](../../docs/speech_generation/reference.md) to learn more about configuration options and limitations.

### Deployment

**Deploying with Docker**

```bash
mkdir models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --source_model luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS --model_repository_path /models --model_name Kokoro-82M-OpenVINO-FP16-OVMS --target_device CPU --task text2speech
```

**Deploying on Bare Metal**

```bat
mkdir c:\models
ovms --rest_port 8000 --source_model luis-castillo/Kokoro-82M-OpenVINO-FP16-OVMS --model_repository_path c:\models --model_name Kokoro-82M-OpenVINO-FP16-OVMS --target_device CPU --task text2speech
```

### Request Generation 

:::{dropdown} **Unary call with curl**


```bash
curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"Kokoro-82M-OpenVINO-FP16-OVMS\", \"voice\": \"af_alloy\", \"input\": \"The quick brown fox jumped over the lazy dog\"}" -o speech.wav
```
:::

:::{dropdown} **Unary call with OpenAI python library**

```python
from pathlib import Path
from openai import OpenAI

prompt = "The quick brown fox jumped over the lazy dog"
filename = "speech.wav"
url="http://localhost:8000/v3"


speech_file_path = Path(__file__).parent / "speech.wav"
client = OpenAI(base_url=url, api_key="not_used")

with client.audio.speech.with_streaming_response.create(
  model="Kokoro-82M-OpenVINO-FP16-OVMS",
  voice="af_alloy",
  input=prompt
) as response:
  response.stream_to_file(speech_file_path)


print("Generation finished")
```
:::

Play speech.wav file to check generated speech.

## Benchmarking speech generation
An asynchronous benchmarking client can be used to access the model server performance with various load conditions. Below are execution examples captured on Intel(R) Core(TM) Ultra 7 258V.

> **Note:** `RTFx` (Real-Time Factor, inverted) is calculated as `generated_audio_duration / generation_time`.
> Values greater than `1.0x` mean faster-than-real-time generation, while values below `1.0x` mean slower-than-real-time.

```console
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/benchmark/v3/requirements.txt
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/benchmark/v3/benchmark.py -o benchmark.py
python benchmark.py --api_url http://localhost:8000/v3/audio/speech --model Kokoro-82M-OpenVINO-FP16-OVMS --batch_size 1 --limit 1000 --request_rate inf --backend text2speech --dataset edinburghcstr/ami --hf-subset ihm --voice af_alloy
Number of documents: 1000
100%|█████████████████████████████████████████████████████████████████████████████████| 1000/1000 [16:37<00:00,  1.00it/s]
Success rate: 100.0%. (1000/1000)
Mean latency: 1506.60 ms
Median latency: 1072.72 ms
Mean RTFx: 3.655x
Median RTFx: 3.899x
```

> **Note:** `RTFx` (Real-Time Factor, inverted) is calculated as `generated_audio_duration / generation_time`.
> Values greater than `1.0x` mean faster-than-real-time generation, while values below `1.0x` mean slower-than-real-time.

## Transcription
### Model preparation
Many variances of Whisper models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/collections/OpenVINO/speech-to-text) and used both for translations and transcriptions endpoints.
In this demo we will use OpenVINO/whisper-large-v3-turbo-fp16-ov, which is a fine-tuned version of Whisper large-v3.

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --task speech2text --source_model OpenVINO/whisper-large-v3-turbo-fp16-ov --model_name OpenVINO/whisper-large-v3-turbo-fp16-ov --model_repository_path /models
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support.
It can be applied using the commands below:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --task speech2text --source_model OpenVINO/whisper-large-v3-turbo-fp16-ov --model_name OpenVINO/whisper-large-v3-turbo-fp16-ov --model_repository_path /models --target_device GPU
```
:::

:::{dropdown} **Deploying on Bare Metal**

If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --task speech2text --source_model OpenVINO/whisper-large-v3-turbo-fp16-ov --model_name OpenVINO/whisper-large-v3-turbo-fp16-ov --model_repository_path models --target_device GPU
```
:::

The default configuration should work in most cases but the parameters can be tuned via OVMS arguments. See the [s2t calculator documentation](../../docs/speech_recognition/reference.md) to learn more about configuration options and limitations.

### Request Generation 
Transcribe the speech.wav file generated in the [Speech generation](#speech-generation) section.

> **Note:** Streaming responses are supported for `audio/transcriptions`. `audio/translations` does not support streaming.

:::{dropdown} **Unary call with cURL**


```bash
curl http://localhost:8000/v3/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@speech.wav" -F model="OpenVINO/whisper-large-v3-turbo-fp16-ov" -F language="en"
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
  model="OpenVINO/whisper-large-v3-turbo-fp16-ov",
  language="en",
  file=audio_file
)

print(transcript.text)
```
```
The quick brown fox jumped over the lazy dog.
```
:::

:::{dropdown} **Streaming call with cURL**

```bash
curl -N http://localhost:8000/v3/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@speech.wav" \
  -F model="OpenVINO/whisper-large-v3-turbo-fp16-ov" \
  -F language="en" \
  -F stream="true"
```

Example streamed chunks (SSE format):
```text
data: {"type":"transcript.text.delta","delta":"The quick ","logprobs":[]}

data: {"type":"transcript.text.delta","delta":"brown fox ","logprobs":[]}

data: {"type":"transcript.text.done","text":"The quick brown fox jumped over the lazy dog.","logprobs":[]}
```
:::

:::{dropdown} **Unary call with sentence timestamps**


```bash
curl http://localhost:8000/v3/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@speech.wav" -F model="OpenVINO/whisper-large-v3-turbo-fp16-ov" -F language="en" -F timestamp_granularities[]="segment"
```
```json
{"text":" A quick brown fox jumped over the lazy dog","segments":[{"text":" A quick brown fox jumped over the lazy dog","start":0.0,"end":3.1399998664855957}]}
```
:::

:::{dropdown} **Unary call with python OpenAI library with sentence timestamps**

```python
from pathlib import Path
from openai import OpenAI

filename = "speech.wav"
url="http://localhost:8000/v3"


speech_file_path = Path(__file__).parent / filename
client = OpenAI(base_url=url, api_key="not_used")

audio_file = open(filename, "rb")
transcript = client.audio.transcriptions.create(
  model="OpenVINO/whisper-large-v3-turbo-fp16-ov",
  language="en",
  response_format="verbose_json",
  timestamp_granularities=["segment"],
  file=audio_file
)

print(transcript.text)
print(transcript.segments)
```
```
 A quick brown fox jumped over the lazy dog
[TranscriptionSegment(id=None, avg_logprob=None, compression_ratio=None, end=3.1399998664855957, no_speech_prob=None, seek=None, start=0.0, temperature=None, text=' A quick brown fox jumped over the lazy dog', tokens=None)]
```
:::

### Word timestamps
If you need word-level timestamps support, export the model with `export_model.py` and enable this feature during export.

Prepare export script and dependencies:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
mkdir -p models
```

Export Speech-to-Text model with word timestamps enabled:
```console
python export_model.py speech2text --source_model openai/whisper-large-v3-turbo --weight-format fp16 --model_name whisper-large-v3-turbo-word-ts --config_file_path models/config.json --model_repository_path models --overwrite_models --enable_word_timestamps
```

:::{dropdown} **Deploying with Docker**

```bash
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --model_path /models/whisper-large-v3-turbo-word-ts --model_name whisper-large-v3-turbo-word-ts
```
:::

:::{dropdown} **Deploying on Bare Metal**

```bat
ovms --rest_port 8000 --model_path models/whisper-large-v3-turbo-word-ts --model_name whisper-large-v3-turbo-word-ts
```
:::

:::{dropdown} **Unary call with cURL (word timestamps)**

```bash
curl http://localhost:8000/v3/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@speech.wav" -F model="whisper-large-v3-turbo-word-ts" -F language="en" -F timestamp_granularities[]="word"
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
url="http://localhost:8000/v3"


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
```console
git clone https://github.com/huggingface/open_asr_leaderboard.git
cd open_asr_leaderboard
```

Download and apply OVMS API compatibility patch:

    curl -L https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/external/open_asr_leaderboard.patch -o ovms_open_asr_leaderboard.patch
    git apply ovms_open_asr_leaderboard.patch

Set OpenAI-compatible endpoint variables for OVMS:
```console
export OPENAI_BASE_URL=http://localhost:8000/v3
export OPENAI_API_KEY="unused"
```

Install dependencies:
```console
pip install -r requirements/requirements.txt -r requirements/requirements-api.txt openai>=1.0.0 torchcodec==0.12
```

Run evaluation example:
```console
PYTHONPATH=. python api/run_eval.py \
  --model_name openai/OpenVINO/whisper-large-v3-turbo-fp16-ov \
  --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
  --max_workers 1 \
  --split test.clean  \
  --dataset "librispeech"
```
Results:
```console
...
Transcribing: 100%|█████████▉| 2617/2620 [12:15<00:00,  5.23it/s]
Transcribing: 100%|█████████▉| 2618/2620 [12:15<00:00,  5.31it/s]
Transcribing: 100%|█████████▉| 2619/2620 [12:16<00:00,  5.41it/s]
Transcribing: 100%|██████████| 2620/2620 [12:16<00:00,  5.20it/s]
Transcribing: 100%|██████████| 2620/2620 [12:16<00:00,  3.56it/s]
Results saved at path: ./results/MODEL_openai-OpenVINO-whisper-large-v3-turbo-fp16-ov_DATASET_hf-audio-esb-datasets-test-only-sorted_librispeech_test.clean.jsonl
WER: 1.97 %
RTFx: 28.87
```

Where:
- WER (Word Error Rate) is the percentage of transcription errors compared to the reference text (substitutions + deletions + insertions). Lower is better.
- RTFx (Real-Time Factor, expressed as speedup) indicates processing speed relative to audio duration. Values above 1 mean faster-than-real-time transcription (for example, 5.16 means about 5.16x real time).

**For Open ASR Leaderboard, run `run_eval.py` with model name prefixed by `openai/` (for example `openai/OpenVINO/whisper-large-v3-turbo-fp16-ov`).**
**OVMS should still be deployed with `--model_name OpenVINO/whisper-large-v3-turbo-fp16-ov` (evaluation script does not include `openai/` prefix in requests).**
You can replace `librispeech` with other datasets supported by the leaderboard configuration. For multilingual models run_eval_ml.py should be used.

## Translation
To test the translations endpoint we first need to prepare an audio file with speech in a language other than English, e.g. Spanish. To generate such a sample, follow the [Speech generation](#speech-generation) section to deploy Kokoro and then run:

```console
curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"Kokoro-82M-OpenVINO-FP16-OVMS\", \"voice\": \"em_alex\", \"input\": \"Madrid es la capital de España\"}" -o speech_spanish.wav
```

### Deployment
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
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --source_model OpenVINO/whisper-large-v3-fp16-ov --model_repository_path /models --model_name OpenVINO/whisper-large-v3-fp16-ov --task speech2text --target_device GPU
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
Translate the speech_spanish.wav file generated above.

:::{dropdown} **Unary call with cURL**


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
