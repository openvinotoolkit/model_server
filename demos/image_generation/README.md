# How to serve Stable Diffusion / FLUX models via OpenAI API {#ovms_demos_image_generation}

This demo shows how to deploy image generation models (Stable Diffusion/Stable Diffusion 3/Stable Diffusion XL/FLUX) in the OpenVINO Model Server using optimized pipelines.
Image generation use case is exposed via [OpenAI API](https://platform.openai.com/docs/api-reference/images/create) `images/generations` endpoints.
That makes it easy to use and efficient especially on on Intel® Xeon®, Intel® Core® processors (including iGPU*), Intel® NPUs* and Intel® discrete GPUs.

> * Untested

> **Note:** This demo was tested on Intel® Xeon®, Intel® Core®, Intel® Arc™ A770 on Ubuntu 22/24, RedHat 9 and Windows 11.

## Prerequisites

**Model preparation** (one of the below):
- preconfigured models from HuggingFaces directly in OpenVINO IR format, list of Intel uploaded models available [here](https://huggingface.co/collections/OpenVINO/image-generation-67697d9952fb1eee4a252aa8))
- or Python 3.9+ with pip and HuggingFace account to download, convert and quantize manually using [Export Models Tool](../common/export_models/README.md)

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**Client**:  Python for using OpenAI client package and Pillow to save image or simply cURL


## Server deployment

:::{dropdown} **Downloading the models directly via OVMS**

> NOTE: Model downloading feature is described in depth in separate documentation page: [Pulling HuggingFaces Models](../../docs/pull_hf_models.md).

This command pulls the `OpenVINO/FLUX.1-schnell-int8-ov` quantized model directly from HuggingFaces and starts the serving. If the model already exists locally, it will skip the downloading and just start the serving.

> NOTE: Optionally, to only download the model and omit the serving part, use `--pull` parameter.

**CPU**
::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
```
mkdir -p models

docker run -d --rm --user $(id -u):$(id -g) -p 8000:8000 -v $(pwd)/models:/models/:rw -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy openvino/model_server:2025.2 --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov
```
:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal
**Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```
mkdir -p models

ovms.exe --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov
```
:::
::::

**GPU**

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
In case you want to use Intel GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` to `docker run` command, use the docker image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:
```bash
mkdir -p models

docker run -d --rm --user $(id -u):$(id -g) --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -p 8000:8000 -v $(pwd)/models:/models/:rw -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy openvino/model_server:2025.2-gpu --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov --target_device GPU
```
:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal

```bash
mkdir -p models

ovms.exe --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov --target_device GPU
```

:::
::::

:::{dropdown} Logs
```
...
Downloading text_encoder/openvino_model.bin (124 MB)
Downloading text_encoder_2/openvino_model.bin (4.8 GB)
Possibly malformed smudge on Windows: see `git lfs help smudge` for more info.
Downloading tokenizer/openvino_detokenizer.bin (617 KB)
Downloading tokenizer/openvino_tokenizer.bin (1.4 MB)
Downloading tokenizer_2/openvino_detokenizer.bin (794 KB)
Downloading tokenizer_2/openvino_tokenizer.bin (794 KB)
Downloading tokenizer_2/spiece.model (792 KB)
Downloading transformer/openvino_model.bin (12 GB)
Possibly malformed smudge on Windows: see `git lfs help smudge` for more info.
Downloading vae_decoder/openvino_model.bin (50 MB)
Downloading vae_encoder/openvino_model.bin (34 MB)
...
```

Your directory structure should look like this:
```
models
`-- OpenVINO
    `-- FLUX.1-schnell-int8-ov
        |-- README.md
        |-- graph.pbtxt
        |-- model_index.json
        |-- scheduler
        |   `-- scheduler_config.json
        |-- text_encoder
        |   |-- config.json
        |   |-- openvino_model.bin
        |   `-- openvino_model.xml
        |-- text_encoder_2
        |   |-- config.json
        |   |-- openvino_model.bin
        |   `-- openvino_model.xml
        |-- tokenizer
        |   |-- merges.txt
        |   |-- openvino_detokenizer.bin
        |   |-- openvino_detokenizer.xml
        |   |-- openvino_tokenizer.bin
        |   |-- openvino_tokenizer.xml
        |   |-- special_tokens_map.json
        |   |-- tokenizer_config.json
        |   `-- vocab.json
        |-- tokenizer_2
        |   |-- openvino_detokenizer.bin
        |   |-- openvino_detokenizer.xml
        |   |-- openvino_tokenizer.bin
        |   |-- openvino_tokenizer.xml
        |   |-- special_tokens_map.json
        |   |-- spiece.model
        |   |-- tokenizer.json
        |   `-- tokenizer_config.json
        |-- transformer
        |   |-- config.json
        |   |-- openvino_model.bin
        |   `-- openvino_model.xml
        |-- vae_decoder
        |   |-- config.json
        |   |-- openvino_model.bin
        |   `-- openvino_model.xml
        `-- vae_encoder
            |-- config.json
            |-- openvino_model.bin
            `-- openvino_model.xml
```
:::

:::


:::{dropdown} **Using export script to download, convert and quantize then start the serving**
Here, the original models in `safetensors` format and the tokenizers will be converted to OpenVINO IR format and optionally quantized to desired precision.
Quantization ensures faster initialization time, better performance and lower memory consumption.
Image generation pipeline parameters will be defined inside the `graph.pbtxt` file.

Download export script (2025.2 and later), install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/main/demos/common/export_models/requirements.txt
mkdir models  # TODO: Change main to 2025.2 release branch
```

Run `export_model.py` script to download and quantize the model:

> **Note:** Before downloading the model, access must be requested. Follow the instructions on the [HuggingFace model page](black-forest-labs/FLUX.1-schnell) to request access. When access is granted, create an authentication token in the HuggingFace account -> Settings -> Access Tokens page. Issue the following command and enter the authentication token. Authenticate via `huggingface-cli login`. 
> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py image_generation --source_model black-forest-labs/FLUX.1-schnell --weight-format int8 --config_file_path models/config.json --model_repository_path models --overwrite_models
```

**GPU**
```console
python export_model.py image_generation --source_model black-forest-labs/FLUX.1-schnell --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models --overwrite_models
```

> **Note:** Change the `--weight-format` to quantize the model to `int8` or `fp16` precision to reduce memory consumption and improve performance, or omit this parameter to keep the original precision.

> **Note:** You can change the model used in the demo, please verify [tested models](https://github.com/openvinotoolkit/openvino.genai/blob/master/tests/python_tests/models/real_models) list.

:::{dropdown} Logs
You should have a model folder like below:
```
models
├── black-forest-labs
│   └── FLUX.1-schnell
│       ├── graph.pbtxt
│       ├── model_index.json
│       ├── scheduler
│       │   └── scheduler_config.json
│       ├── text_encoder
│       │   ├── config.json
│       │   ├── openvino_model.bin
│       │   └── openvino_model.xml
│       ├── text_encoder_2
│       │   ├── config.json
│       │   ├── openvino_model.bin
│       │   └── openvino_model.xml
│       ├── tokenizer
│       │   ├── merges.txt
│       │   ├── openvino_detokenizer.bin
│       │   ├── openvino_detokenizer.xml
│       │   ├── openvino_tokenizer.bin
│       │   ├── openvino_tokenizer.xml
│       │   ├── special_tokens_map.json
│       │   ├── tokenizer_config.json
│       │   └── vocab.json
│       ├── tokenizer_2
│       │   ├── openvino_detokenizer.bin
│       │   ├── openvino_detokenizer.xml
│       │   ├── openvino_tokenizer.bin
│       │   ├── openvino_tokenizer.xml
│       │   ├── special_tokens_map.json
│       │   ├── spiece.model
│       │   ├── tokenizer_config.json
│       │   └── tokenizer.json
│       ├── transformer
│       │   ├── config.json
│       │   ├── openvino_model.bin
│       │   └── openvino_model.xml
│       ├── vae_decoder
│       │   ├── config.json
│       │   ├── openvino_model.bin
│       │   └── openvino_model.xml
│       └── vae_encoder
│           ├── config.json
│           ├── openvino_model.bin
│           └── openvino_model.xml
└── config.json
```
:::

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [Image Generation calculator documentation](../../docs/image_generation/reference.md) to learn more about configuration options.

### Server Deployment

**Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models:ro openvino/model_server:2025.2 --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path /models/black-forest-labs/FLUX.1-schnell
```
:::

**Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

:::{tab-item} Bare metal (Windows)
:sync: bare-metal
```bash
ovms.exe --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path ./models/black-forest-labs/FLUX.1-schnell
```
:::

::::

**GPU**

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:
```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/workspace:ro openvino/model_server:2025.2-gpu --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path /models/black-forest-labs/FLUX.1-schnell
```

:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal
```bash
ovms.exe --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path ./models/black-forest-labs/FLUX.1-schnell
```
:::

::::


:::

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
 "black-forest-labs/FLUX.1-schnell" :
 {
  "model_version_status": [
   {
    "version": "1",
    "state": "AVAILABLE",
    "status": {
     "error_code": "OK",
     "error_message": "OK"
    }
   }
  ]
 }
}
```

## Request Generation

A single servable exposes all 3 endpoints:
- text to image: `images/generations` DONE
- image to image: `images/edits` TODO
- inpainting: `images/edits` with `mask` field TODO

All requests are processed in unary format, with no streaming capabilities.

### Requesting images/generations API using cURL 

Linux
```bash
curl http://localhost:8000/v3/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-schnell",
    "prompt": "three cats",
    "num_inference_steps": 2,
    "size": "512x512"
  }'| jq -r '.data[0].b64_json' | base64 --decode > output.png
```

Windows Powershell
```powershell
(Invoke-WebRequest -Uri "http://localhost:8000/v3/images/generations" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "black-forest-labs/FLUX.1-schnell", "prompt": "three cats", "num_inference_steps": 50, "size": "512x512"}').Content
```
TODO: Save to disk

Windows Command Prompt
```bat
curl -s http://localhost:8000/v3/images/generations -H "Content-Type: application/json" -d "{\"model\": \"black-forest-labs/FLUX.1-schnell\", \"prompt\": \"three cats\", \"num_inference_steps\": 50, \"size\": \"512x512\"}"
```
TODO: Save to disk

Expected Response
```json
{
  "data": [
    {
      "b64_json": "..."
    }
  ]
}
```

The commands will have the generated image saved in output.png.

![output](./output.png)


### Requesting image generation with OpenAI Python package

The image generation/edit endpoints are compatible with OpenAI client:

Install the client library:
```console
pip3 install openai pillow
```

```console
pip3 install openai
```
```python
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image

client = OpenAI(
    base_url="http://localhost:8000/v3",
    api_key="unused"
)

response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell",
            prompt="three cats",
            n=1,
            extra_body={
                "rng_seed": 42,
                "num_inference_steps": 3
            }
        )
base64_image = response.data[0].b64_json

image_data = base64.b64decode(base64_image)
image = Image.open(BytesIO(image_data))
image.save('out.png')

```

Output file (`out.png`):
![out2](./output2.png)


## Benchmarking text generation with high concurrency

TODO

## Testing the model accuracy over serving API

TODO


## References
- [Image Generation API](../../docs/model_server_rest_api_image_generation.md)
- [Writing client code](../../docs/clients_genai.md)
- [Image Generation calculator reference](../../docs/image_generation/reference.md)
