How to serve Stable Diffusion / FLUX models via OpenAI API  {#ovms_demos_image_generation}
==========================================================

This demo shows how to deploy image generation models (Stable Diffusion/Stable Diffusion 3/Stable Diffusion XL/FLUX) in the OpenVINO Model Server using optimized pipelines.
Image generation use case is exposed via `OpenAI API <https://platform.openai.com/docs/api-reference/images/create>`_ ``images/generations`` endpoints.
That makes it easy to use and efficient especially on Intel® Xeon®, Intel® Core® processors (including iGPU*), Intel® NPUs* and Intel® discrete GPUs.

.. note::
   This demo was tested on Intel® Xeon®, Intel® Core®, Intel® Arc™ A770 on Ubuntu 22/24, RedHat 9 and Windows 11.

Prerequisites
-------------

**Model preparation** (one of the below):

- Preconfigured models from HuggingFaces directly in OpenVINO IR format, list of Intel uploaded models available `here <https://huggingface.co/collections/OpenVINO/image-generation-67697d9952fb1eee4a252aa8>`_
- Or Python 3.9+ with pip and HuggingFace account to download, convert and quantize manually using :doc:`../common/export_models/README`

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the :doc:`../../docs/deploying_server_baremetal`

**Client**: Python for using OpenAI client package and Pillow to save image or simply cURL

Downloading the models directly via OVMS
----------------------------------------

.. note::
   Model downloading feature is described in depth in separate documentation page: :doc:`../../docs/pull_hf_models`.

This command pulls the ``OpenVINO/FLUX.1-schnell-int8-ov`` quantized model directly from HuggingFaces and starts the serving. If the model already exists locally, it will skip the downloading and just start the serving.

.. note::
   Optionally, to only download the model and omit the serving part, use ``--pull`` parameter.

**CPU**

.. tabs::

   .. tab:: Docker (Linux)

      .. code-block:: bash

         mkdir -p models

         docker run -d --rm --user $(id -u):$(id -g) -p 8000:8000 -v $(pwd)/models:/models/:rw -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy openvino/model_server:2025.2 --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov

   .. tab:: Bare metal (Windows)

      **Deploying on Bare Metal**

      Assuming you have unpacked model server package, make sure to:

      - **On Windows**: run ``setupvars`` script
      - **On Linux**: set ``LD_LIBRARY_PATH`` and ``PATH`` environment variables

      as mentioned in :doc:`../../docs/deploying_server_baremetal`, in every new shell that will start OpenVINO Model Server.

      .. code-block:: bash

         mkdir -p models

         ovms.exe --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov

**GPU**

.. tabs::

   .. tab:: Docker (Linux)

      In case you want to use Intel GPU device to run the generation, add extra docker parameters ``--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`` to ``docker run`` command, use the docker image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
      It can be applied using the commands below:

      .. code-block:: bash

         mkdir -p models

         docker run -d --rm --user $(id -u):$(id -g) --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -p 8000:8000 -v $(pwd)/models:/models/:rw -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy openvino/model_server:2025.2-gpu --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov --target_device GPU

   .. tab:: Bare metal (Windows)

      .. code-block:: bash

         mkdir -p models

         ovms.exe --rest_port 8000 --model_repository_path /models/ --task image_generation --source_model OpenVINO/FLUX.1-schnell-int8-ov --target_device GPU

.. dropdown:: Logs

   .. code-block:: text

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

   Your directory structure should look like this:

   .. code-block:: text

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

Using export script to download, convert and quantize then start the serving
---------------------------------------------------------------------------

Here, the original models in ``safetensors`` format and the tokenizers will be converted to OpenVINO IR format and optionally quantized to desired precision.
Quantization ensures faster initialization time, better performance and lower memory consumption.
Image generation pipeline parameters will be defined inside the ``graph.pbtxt`` file.

Download export script (2025.2 and later), install its dependencies and create directory for the models:

.. code-block:: console

   curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/main/demos/common/export_models/export_model.py -o export_model.py
   pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/main/demos/common/export_models/requirements.txt
   mkdir models  # TODO: Change main to 2025.2 release branch

Run ``export_model.py`` script to download and quantize the model:

.. note::
   Before downloading the model, access must be requested. Follow the instructions on the `HuggingFace model page <https://huggingface.co/black-forest-labs/FLUX.1-schnell>`_ to request access. When access is granted, create an authentication token in the HuggingFace account -> Settings -> Access Tokens page. Issue the following command and enter the authentication token. Authenticate via ``huggingface-cli login``.

.. note::
   The users in China need to set environment variable ``HF_ENDPOINT="https://hf-mirror.com"`` before running the export script to connect to the HF Hub.

**CPU**

.. code-block:: console

   python export_model.py image_generation --source_model black-forest-labs/FLUX.1-schnell --weight-format int8 --config_file_path models/config.json --model_repository_path models --overwrite_models

**GPU**

.. code-block:: console

   python export_model.py image_generation --source_model black-forest-labs/FLUX.1-schnell --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models --overwrite_models

.. note::
   Change the ``--weight-format`` to quantize the model to ``int8`` or ``fp16`` precision to reduce memory consumption and improve performance, or omit this parameter to keep the original precision.

.. note::
   You can change the model used in the demo, please verify `tested models <https://github.com/openvinotoolkit/openvino.genai/blob/master/tests/python_tests/models/real_models>`_ list.

.. dropdown:: Logs

   You should have a model folder like below:

   .. code-block:: text

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

The default configuration should work in most cases but the parameters can be tuned via ``export_model.py`` script arguments. Run the script with ``--help`` argument to check available parameters and see the :doc:`../../docs/image_generation/reference` to learn more about configuration options.

Server Deployment
-----------------

**Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:

.. tabs::

   .. tab:: Docker (Linux)

      .. code-block:: bash

         docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models:ro openvino/model_server:2025.2 --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path /models/black-forest-labs/FLUX.1-schnell

   .. tab:: Bare metal (Windows)

      .. code-block:: bash

         ovms.exe --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path ./models/black-forest-labs/FLUX.1-schnell

**GPU**

.. tabs::

   .. tab:: Docker (Linux)

      .. code-block:: bash

         docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/workspace:ro openvino/model_server:2025.2-gpu --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path /models/black-forest-labs/FLUX.1-schnell

   .. tab:: Bare metal (Windows)

      .. code-block:: bash

         ovms.exe --rest_port 8000 --model_name black-forest-labs/FLUX.1-schnell --model_path ./models/black-forest-labs/FLUX.1-schnell

Readiness Check
---------------

Wait for the model to load. You can check the status with a simple command:

.. code-block:: console

   curl http://localhost:8000/v1/config

.. code-block:: json

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

Request Generation
------------------

A single servable exposes all 3 endpoints:

- text to image: ``images/generations`` DONE
- image to image: ``images/edits`` TODO
- inpainting: ``images/edits`` with ``mask`` field TODO

All requests are processed in unary format, with no streaming capabilities.

Requesting images/generations API using cURL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux**

.. code-block:: bash

   curl http://localhost:8000/v3/images/generations \
     -H "Content-Type: application/json" \
     -d '{
       "model": "black-forest-labs/FLUX.1-schnell",
       "prompt": "three cats",
       "num_inference_steps": 2,
       "size": "512x512"
     }'| jq -r '.data[0].b64_json' | base64 --decode > output.png

**Windows Powershell**

.. code-block:: powershell

   (Invoke-WebRequest -Uri "http://localhost:8000/v3/images/generations" `
    -Method POST `
    -Headers @{ "Content-Type" = "application/json" } `
    -Body '{"model": "black-forest-labs/FLUX.1-schnell", "prompt": "three cats", "num_inference_steps": 50, "size": "512x512"}').Content

TODO: Save to disk

**Windows Command Prompt**

.. code-block:: bat

   curl -s http://localhost:8000/v3/images/generations -H "Content-Type: application/json" -d "{\"model\": \"black-forest-labs/FLUX.1-schnell\", \"prompt\": \"three cats\", \"num_inference_steps\": 50, \"size\": \"512x512\"}"

TODO: Save to disk

Expected Response

.. code-block:: json

   {
     "data": [
       {
         "b64_json": "..."
       }
     ]
   }

The commands will have the generated image saved in output.png.

.. image:: ./output.png

Requesting image generation with OpenAI Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The image generation/edit endpoints are compatible with OpenAI client:

Install the client library:

.. code-block:: console

   pip3 install openai pillow

.. code-block:: python

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

Output file (``out.png``):

.. image:: ./output2.png

Benchmarking text generation with high concurrency
-------------------------------------------------

TODO

Testing the model accuracy over serving API
-------------------------------------------

TODO

References
----------

- :doc:`../../docs/model_server_rest_api_image_generation`
- :doc:`../../docs/clients_genai`
- :doc:`../../docs/image_generation/reference`
