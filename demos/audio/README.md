# How to serve audio models via OpenAI API {#ovms_demos_continuous_batching_vlm}

This demo shows how to deploy audio in the OpenVINO Model Server.
Speech generation and speech recognition use cases are exposed via OpenAI API `audio/speech` end `audio/transcriptions` endpoint.

## Prerequisites

**OVMS version 2025.4** This demo require version 2025.4 or newer.

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package


## Speech generation
### Model preparation
Only supported model for speech generation use case is microsoft/speecht5_tts which is outside OpenVINO organization and needs convertion to IR format.

Specific OVMS pull mode example for models outside of OpenVINO organization is described in section `## Pulling models outside of OpenVINO organization` in the [Ovms pull mode](https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_hf_models.md)

Or you can use the python export_model.py script described below.

Here, the original TTS model will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
Execution parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py text_to_speech --source_model microsoft/speecht5_tts --weight-format int4 --pipeline_type VLM --model_name microsoft/speecht5_tts --config_file_path models/config.json --model_repository_path models  --overwrite_models
```

**GPU**
```console
python export_model.py text_generation --source_model microsoft/speecht5_tts --weight-format int4 --pipeline_type VLM --model_name microsoft/speecht5_tts --config_file_path models/config.json --model_repository_path models --overwrite_models --target_device GPU
```

> **Note:** Change the `--weight-format` to quantize the model to `fp16` or `int8` precision to reduce memory consumption and improve performance.


You should have a model folder like below:
```
models/
├── config.json
└── speecht5_tts
    ├── added_tokens.json
    ├── config.json
    ├── generation_config.json
    ├── graph.pbtxt
    ├── openvino_decoder_model.bin
    ├── openvino_decoder_model.xml
    ├── openvino_detokenizer.bin
    ├── openvino_detokenizer.xml
    ├── openvino_encoder_model.bin
    ├── openvino_encoder_model.xml
    ├── openvino_postnet.bin
    ├── openvino_postnet.xml
    ├── openvino_tokenizer.bin
    ├── openvino_tokenizer.xml
    ├── openvino_vocoder.bin
    ├── openvino_vocoder.xml
    ├── preprocessor_config.json
    ├── special_tokens_map.json
    ├── spm_char.model
    └── tokenizer_config.json
```

### Request Generation 

:::{dropdown} **Unary call with curl using image url**


```bash
curl http://localhost:8000/v3/audio/speech -H "Content-Type: application/json" -d "{\"model\": \"microsoft/speecht5_tts\", \"input\": \"The quick brown fox jumped over the lazy dog.\"}" -o speech.wav
```
:::

:::{dropdown} **Unary call with python requests library**

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

speech.wav file contains generated speech.
