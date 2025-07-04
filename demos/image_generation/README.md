# Image Generation with OpenAI API {#ovms_demos_image_generation}

This demo shows how to deploy image generation models (Stable Diffusion/Stable Diffusion 3/Stable Diffusion XL/FLUX) in the OpenVINO Model Server.
Image generation pipeline is exposed via [OpenAI API](https://platform.openai.com/docs/api-reference/images/create) `images/generations` endpoints.

> **Note:** This demo was tested on Intel® Xeon®, Intel® Core®, Intel® Arc™ A770, Intel® Arc™ B580 on Ubuntu 22/24, RedHat 9 and Windows 11.

## Prerequisites

**RAM/vRAM** Select model size and precision according to your hardware capabilities (RAM/vRAM). Request resolution plays significant role in memory consumption, so the higher resolution you request, the more RAM/vRAM is required.

**Model preparation** (one of the below):
- preconfigured models from HuggingFaces directly in OpenVINO IR format, list of Intel uploaded models available [here](https://huggingface.co/collections/OpenVINO/image-generation-67697d9952fb1eee4a252aa8))
- or Python 3.9+ with pip and HuggingFace account to download, convert and quantize manually using [Export Models Tool](../common/export_models/README.md)

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**Client**:  Python for using OpenAI client package and Pillow to save image or simply cURL


## Option 1. Downloading the models directly via OVMS

> **NOTE:** Model downloading feature is described in depth in separate documentation page: [Pulling HuggingFaces Models](../../docs/pull_hf_models.md).

This command pulls the `OpenVINO/FLUX.1-schnell-int4-ov` quantized model directly from HuggingFaces and starts the serving. If the model already exists locally, it will skip the downloading and immediately start the serving.

> **NOTE:** Optionally, to only download the model and omit the serving part, use `--pull` parameter.

### CPU

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
Start docker container:
```bash
mkdir -p models

docker run -d --rm --user $(id -u):$(id -g) -p 8000:8000 -v $(pwd)/models:/models/:rw \
  -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
  openvino/model_server:2025.2 \
    --rest_port 8000 \
    --model_repository_path /models/ \
    --task image_generation \
    --source_model OpenVINO/FLUX.1-schnell-int4-ov
```
:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.


```bat
mkdir models

ovms --rest_port 8000 ^
  --model_repository_path ./models/ ^
  --task image_generation ^
  --source_model OpenVINO/FLUX.1-schnell-int4-ov
```
:::

::::

### GPU

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
In case you want to use Intel GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` to `docker run` command, use the docker image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:
```bash
mkdir -p models

docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models/:rw \
  --user $(id -u):$(id -g) --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
  -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
  openvino/model_server:2025.2-gpu \
    --rest_port 8000 \
    --model_repository_path /models/ \
    --task image_generation \
    --source_model OpenVINO/FLUX.1-schnell-int4-ov \
    --target_device GPU
```
:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
mkdir models

ovms --rest_port 8000 ^
  --model_repository_path ./models/ ^
  --task image_generation ^
  --source_model OpenVINO/FLUX.1-schnell-int4-ov ^
  --target_device GPU
```
:::

::::


### NPU or mixed device

Image generation endpoints consist of 3 steps: text encoding, denoising and vae decoder. It is possible to select device for each step separately. In this example, we will use NPU for text encoding and denoising, and GPU for vae decoder. This is useful when the model is too large to fit into NPU memory, but the NPU can still be used for the first two steps.

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
In case you want to use Intel NPU device to run the generation, add extra docker parameters `--device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` to `docker run` command, use the docker image with NPU support. Export the models with precision matching the NPU capacity and adjust pipeline configuration.
In this specific case, we also need to use `--device /dev/dri`, because we also use GPU.

> **NOTE:** The NPU device requires the pipeline to be reshaped to static shape, this is why the `--resolution` parameter is used to define the input resolution.

> **NOTE:** This feature will be available in 2025.3 and later releases, so until next release, it is required to build the model server from source from the `main` branch.


It can be applied using the commands below:
```bash
mkdir -p models

docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models/:rw \
  --user $(id -u):$(id -g) --device /dev/accel --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
  -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
  openvino/model_server:latest-gpu \
    --rest_port 8000 \
    --model_repository_path /models/ \
    --task image_generation \
    --source_model OpenVINO/FLUX.1-schnell-int4-ov \
    --target_device 'NPU NPU GPU' \
    --resolution 512x512
```
:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal


```bat
mkdir models

ovms --rest_port 8000 ^
  --model_repository_path ./models/ ^
  --task image_generation ^
  --source_model OpenVINO/FLUX.1-schnell-int4-ov ^
  --target_device 'NPU NPU GPU' ^
  --resolution 512x512
```
:::

::::


## Option 2. Using export script to download, convert and quantize then start the serving
Here, the original models in `safetensors` format and the tokenizers will be converted to OpenVINO IR format and optionally quantized to desired precision.
Quantization ensures faster initialization time, better performance and lower memory consumption.
Image generation pipeline parameters will be defined inside the `graph.pbtxt` file.

Download export script (2025.2 and later), install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** Before downloading the model, access must be requested. Follow the instructions on the [HuggingFace model page](https://huggingface.co/black-forest-labs/FLUX.1-schnell) to request access. When access is granted, create an authentication token in the HuggingFace account -> Settings -> Access Tokens page. Issue the following command and enter the authentication token. Authenticate via `huggingface-cli login`. 

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

### Export model for CPU
```console
python export_model.py image_generation \
  --source_model black-forest-labs/FLUX.1-schnell \
  --weight-format int4 \
  --config_file_path models/config.json \
  --model_repository_path models \
  --overwrite_models
```

### Export model for GPU
```console
python export_model.py image_generation \
  --source_model black-forest-labs/FLUX.1-schnell \
  --weight-format int4 \
  --target_device GPU \
  --config_file_path models/config.json \
  --model_repository_path models \
  --overwrite_models
```

### Export model for NPU or mixed device

Image generation endpoints consist of 3 steps: text encoding, denoising and vae decoder. It is possible to select device for each step separately. In this example, we will use NPU for text encoding and denoising, and GPU for vae decoder. This is useful when the model is too large to fit into NPU memory, but the NPU can still be used for the first two steps.

> **NOTE:** The NPU device requires the pipeline to be reshaped to static shape, this is why the `--resolution` parameter is used to define the input resolution.

> **NOTE:** This feature will be available in 2025.3 and later releases, so until next release, it is required to use export script from the `main` branch.


```console
python export_model.py image_generation \
  --source_model black-forest-labs/FLUX.1-schnell \
  --weight-format int4 \
  --target_device 'NPU NPU GPU' \
  --resolution '512x512' \
  --config_file_path models/config.json \
  --model_repository_path models \
  --overwrite_models
```

> **Note:** Change the `--weight-format` to quantize the model to `int8`, `fp16` or `int4` precision to reduce memory consumption and improve performance, or omit this parameter to keep the original precision.

> **Note:** You can change the model used in the demo, please verify [tested models](https://github.com/openvinotoolkit/openvino.genai/blob/master/tests/python_tests/models/real_models) list.


The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [Image Generation calculator documentation](../../docs/image_generation/reference.md) to learn more about configuration options.

### Server Deployment

**Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**  

Running this command starts the container with CPU only target device:

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker

Start docker container:
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models:ro \
  openvino/model_server:2025.2 \
    --rest_port 8000 \
    --model_name OpenVINO/FLUX.1-schnell-int4-ov \
    --model_path /models/black-forest-labs/FLUX.1-schnell
```
:::


:::{tab-item} Bare metal (Windows)
:sync: bare-metal

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

```bat
ovms --rest_port 8000 ^
  --model_name OpenVINO/FLUX.1-schnell-int4-ov ^
  --model_path ./models/black-forest-labs/FLUX.1-schnell
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
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro \
  --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
  openvino/model_server:2025.2-gpu \
    --rest_port 8000 \
    --model_name OpenVINO/FLUX.1-schnell-int4-ov \
    --model_path /models/black-forest-labs/FLUX.1-schnell
```

:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 ^
  --model_name OpenVINO/FLUX.1-schnell-int4-ov ^
  --model_path ./models/black-forest-labs/FLUX.1-schnell
```
:::

::::

**NPU or mixed device**  

This feature will be available in 2025.3 and later releases. Until then, please build the model server from source from the `main` branch.

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker

In case you want to use NPU device to run the generation, add extra docker parameters `--device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with NPU support. Export the models with precision matching the NPU capacity and adjust pipeline configuration.
In this specific case, we also need to use `--device /dev/dri`, because we also use GPU.

It can be applied using the commands below:
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro \
  --device /dev/accel --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
  openvino/model_server:latest-gpu \
    --rest_port 8000 \
    --model_name OpenVINO/FLUX.1-schnell-int4-ov \
    --model_path /models/black-forest-labs/FLUX.1-schnell
```

:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or NPU (it's defined in `config.json`). If you run on NPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 ^
  --model_name OpenVINO/FLUX.1-schnell-int4-ov ^
  --model_path ./models/black-forest-labs/FLUX.1-schnell
```
:::

::::


## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
 "OpenVINO/FLUX.1-schnell-int4-ov" :
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

A single servable exposes following endpoints:
- text to image: `images/generations`

Endpoints unsupported for now:
- image to image: `images/edits` 
- inpainting: `images/edits` with `mask` field

All requests are processed in unary format, with no streaming capabilities.

### Requesting images/generations API using cURL 

Linux
```bash
curl http://localhost:8000/v3/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenVINO/FLUX.1-schnell-int4-ov",
    "prompt": "three cute cats sitting on a bench",
    "rng_seed": 45,
    "num_inference_steps": 3,
    "size": "512x512"
  }'| jq -r '.data[0].b64_json' | base64 --decode > output.png
```

Windows Powershell
```powershell
$response = Invoke-WebRequest -Uri "http://localhost:8000/v3/images/generations" `
    -Method POST `
    -Headers @{ "Content-Type" = "application/json" } `
    -Body '{"model": "OpenVINO/FLUX.1-schnell-int4-ov", "prompt": "three cute cats sitting on a bench", "rng_seed": 45, "num_inference_steps": 3, "size": "512x512"}'

$base64 = ($response.Content | ConvertFrom-Json).data[0].b64_json

[IO.File]::WriteAllBytes('output.png', [Convert]::FromBase64String($base64))
```

Windows Command Prompt
```bat
curl http://localhost:8000/v3/images/generations ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"OpenVINO/FLUX.1-schnell-int4-ov\", \"prompt\": \"three cute cats sitting on a bench\", \"rng_seed\": 45, \"num_inference_steps\": 3, \"size\": \"512x512\"}"
```


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
            model="OpenVINO/FLUX.1-schnell-int4-ov",
            prompt="three cute cats sitting on a bench",
            extra_body={
                "rng_seed": 60,
                "size": "512x512",
                "num_inference_steps": 3
            }
        )
base64_image = response.data[0].b64_json

image_data = base64.b64decode(base64_image)
image = Image.open(BytesIO(image_data))
image.save('output2.png')

```

Output file (`output2.png`):  
![output2](./output2.png)




## References
- [Image Generation API](../../docs/model_server_rest_api_image_generation.md)
- [Writing client code](../../docs/clients_genai.md)
- [Image Generation calculator reference](../../docs/image_generation/reference.md)
