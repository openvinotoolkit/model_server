# How to serve audio models via OpenAI API {#ovms_demos_audio}

This demo shows how to deploy audio models in the OpenVINO Model Server.
Speech generation and speech recognition models are exposed via OpenAI API `audio/speech`, `audio/transcriptions` and `audio/translations` endpoints.
Speech-to-text streaming responses are supported for `audio/transcriptions` endpoint.

Check supported [Speech Recognition Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) and [Speech Generation Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models).

## Prerequisites

**OVMS version 2025.4** This demo requires version 2025.4 or nightly release.

**Model preparation**: Python 3.10 or higher with pip

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**Client**: curl or Python for using OpenAI client package

## Speech generation
### Prepare speaker embeddings
When generating speech you can use default speaker voice or you can prepare your own speaker embedding file. Here you can see how to do it with downloaded file from online repository, but you can try with your own speech recording as well:
```console
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/audio/requirements.txt
mkdir audio_samples
curl --create-dirs "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0032_8k.wav" -o audio_samples/audio.wav
curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/audio/create_speaker_embedding.py -o models/speakers/create_speaker_embedding.py
python models/speakers/create_speaker_embedding.py audio_samples/audio.wav models/speakers/voice1.bin
```

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
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.
> **Note:** Exporting `microsoft/speecht5_tts` model requires Python 3.10

**CPU**
```console
python export_model.py text2speech --source_model microsoft/speecht5_tts --weight-format fp16 --model_name microsoft/speecht5_tts --config_file_path models/config.json --model_repository_path models --overwrite_models --vocoder microsoft/speecht5_hifigan --speaker_name voice1 --speaker_path models/speakers/voice1.bin
```

> **Note:** Change the `--weight-format` to quantize the model to `int8` precision to reduce memory consumption and improve performance.
> **Note:** `speaker_name` and `speaker_path` may be omitted if the default model voice is sufficient

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
ovms --rest_port 8000 --model_path models/microsoft/speecht5_tts --model_name microsoft/speecht5_tts
```

### Request Generation 

:::{dropdown} **Unary call with curl with default voice**


```bash
curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"microsoft/speecht5_tts\", \"input\": \"The quick brown fox jumped over the lazy dog\"}" -o speech.wav
```
:::

:::{dropdown} **Unary call with OpenAI python library with default voice**

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
  voice=None,
  input=prompt
) as response:
  response.stream_to_file(speech_file_path)


print("Generation finished")
```
:::

:::{dropdown} **Unary call with curl**


```bash
curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"microsoft/speecht5_tts\", \"voice\":\"voice1\", \"input\": \"The quick brown fox jumped over the lazy dog\"}" -o speech.wav
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
  model="microsoft/speecht5_tts",
  voice="voice1",
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
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/benchmark/v3/requirements.txt
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/benchmark/v3/benchmark.py -o benchmark.py
python benchmark.py --api_url http://localhost:8000/v3/audio/speech --model microsoft/speecht5_tts --batch_size 1 --limit 100 --request_rate inf --backend text2speech --dataset edinburghcstr/ami --hf-subset ihm --tokenizer OpenVINO/whisper-large-v3-turbo-fp16-ov --trust-remote-code True
Number of documents: 100
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [01:58<00:00,  1.19s/it]
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

> **Note:** Sentence timestamps are supported via `timestamp_granularities[]=segment` in transcription requests.

The default configuration should work in most cases but the parameters can be tuned via OVMS arguments. See the [s2t calculator documentation](../../docs/speech_recognition/reference.md) to learn more about configuration options and limitations.

### Request Generation 
Transcript file that was previously generated with audio/speech endpoint.

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
To test translations endpoint we first need to prepare audio file with speech in language other than English, e.g. Spanish. To generate such sample we will use finetuned version of microsoft/speecht5_tts model.

**Deploying with Docker**

```bash
mkdir -p models

python export_model.py text2speech --source_model Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish --weight-format fp16 --model_name speecht5_tts_spanish --config_file_path models/config.json --model_repository_path models --overwrite_models --vocoder microsoft/speecht5_hifigan

docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --model_path /models/speecht5_tts_spanish --model_name speecht5_tts_spanish

curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"speecht5_tts_spanish\", \"input\": \"Madrid es la capital de España\"}" -o speech_spanish.wav
```

**Deploying on Bare Metal**

```bat
mkdir models

python export_model.py text2speech --source_model Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish --weight-format fp16 --model_name speecht5_tts_spanish --config_file_path models/config.json --model_repository_path models --overwrite_models --vocoder microsoft/speecht5_hifigan

ovms --rest_port 8000 --model_path models/speecht5_tts_spanish --model_name speecht5_tts_spanish

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
Transcript and translate file that was previously generated with audio/speech endpoint.

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
