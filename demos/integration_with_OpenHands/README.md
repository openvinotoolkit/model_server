# OpenHands with OpenVINO Model Server {#ovms_demos_integration_with_openhands}

## Description

[OpenHands](https://github.com/All-Hands-AI/OpenHands) is an open-source software engineering agent that can reason over multi-step tasks, write code, run commands in isolated sandboxes, and inspect the results. This demo configures OpenHands to use a model served by [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) through the OpenAI-compatible REST API.

---

## Setup

### Prerequisites

Requirements:
* Linux host with x86_64 architecture
* [Docker Engine](https://docs.docker.com/engine/) and the Docker Compose v2 plugin
* Enough RAM or GPU memory for the chosen model, plus disk space for the model and container images. The default 27B model requires substantially more memory than smaller models.
* Access to `ghcr.io` to download the OpenHands Agent Canvas image
* An Intel GPU supported by OpenVINO for the optional GPU deployment
* A Hugging Face token in `HF_TOKEN` when the selected model requires authentication

Clone the repository and enter the demo directory:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/integration_with_OpenHands
```

## Step 1: Start OVMS and OpenHands

Compose starts both services, and OVMS downloads the default model, `OpenVINO/Qwen3.8-27B-int8-ov`, into a local cache. Other OpenVINO text-generation models from the [OpenVINO model collection](https://huggingface.co/OpenVINO) can also be used.

Set the host user and group IDs so OVMS can write to the model cache:

```bash
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export MODEL_CACHE_DIR="${HOME}/ovms-openhands/models"
mkdir -p "$MODEL_CACHE_DIR"
```

If the default model does not fit in available memory, try a smaller model such as `OpenVINO/gpt-oss-20b-int4-ov` or `OpenVINO/Qwen3-8b-int8-ov`. Set its Hugging Face ID and a local name for OVMS before starting Compose:

```bash
export MODEL_ID="OpenVINO/gpt-oss-20b-int4-ov"
export LOCAL_NAME="$(basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"
```

For gated models, set `HF_TOKEN` in your shell before starting Compose. The token is passed to the OVMS container; avoid putting it in the Compose file or shell history.

By default, OVMS selects a target device from those available inside its container. Start without passing GPU devices to the container:

```bash
docker compose -f compose.yaml up -d
```

To make an Intel GPU available to OVMS, grant the container access to the render device and use the GPU override:

```bash
export GPU_DEVICE_GID="$(stat -c '%g' /dev/dri/render* | head -n 1)"
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

To select a device explicitly, export `TARGET_DEVICE` (for example, `CPU` or `GPU`) before starting Compose. Unset it to let OVMS choose automatically. The GPU override is still required for GPU access.

The default published ports are `8000` for OVMS and `3000` for OpenHands. To change them, export `OVMS_REST_PORT` and `OPENHANDS_PORT` before starting Compose. `MODEL_CACHE_DIR` changes the cache location and `OVMS_IMAGE` overrides the default `openvino/model_server:weekly` image. OpenHands starts while OVMS downloads and loads the model.

## Step 2: Verify OVMS

Check that the model is available:

```bash
curl http://localhost:${OVMS_REST_PORT:-8000}/v1/config
```

The response should contain `"model_status": "AVAILABLE"`. Send a request through the same OpenAI-compatible endpoint used by OpenHands:

```bash
curl -X POST http://localhost:${OVMS_REST_PORT:-8000}/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-int8-ov",
    "messages": [{"role": "user", "content": "Say this is a test"}],
    "max_tokens": 10
  }'
```

The default local model name is `qwen3.8-27b-int8-ov`. For another model, use its configured `LOCAL_NAME` in the request.

## Step 3: Start an OpenHands Task

Open [http://localhost:3000/canvas](http://localhost:3000/canvas), or use the configured `OPENHANDS_PORT`. OpenHands Agent Canvas 1.46.0 stores its settings and conversations in a Docker volume. Settings from the previous 0.59 demo are not imported, and the existing `.openhands` directory is left unchanged.

On a fresh installation, the `openhands-init` service creates and activates an `ovms` LLM profile with native tool calling enabled. Select the OpenHands agent during first-time setup. Existing LLM profiles are preserved; edit the active profile in **Settings** > **LLM** if you change the model. For the example model, the profile uses:

* **Model:** `openai/qwen3.8-27b-int8-ov`
* **Base URL:** `http://ovms-llm:8000/v3`
* **API Key:** `unused`

For another model, use its configured local model name. The backend must be able to reach the base URL; `localhost` inside the OpenHands container does not refer to OVMS.

Create a task such as:

```text
Create a Python function that calculates the factorial of a number and run a test.
```

OpenHands should write the implementation and execute the test inside the Agent Canvas container. Choose a model that supports tool calling. Conversations share the container environment; this deployment does not create a separate sandbox container for each conversation. Do not mount sensitive host directories into it.

If the UI or backend does not start, inspect its logs:

```bash
docker compose -f compose.yaml logs openhands openhands-init
```

> The OVMS base URL is an internal Docker network address. Use `http://localhost:8000/v3` only for clients running directly on the host.

To confirm that requests reach OVMS, run:

```bash
docker compose -f compose.yaml logs -f ovms-llm
```

## Container Management

Use the Compose configuration for routine operations:

```bash
docker compose -f compose.yaml ps
docker compose -f compose.yaml logs -f
docker compose -f compose.yaml restart
docker compose -f compose.yaml down
```

For a GPU deployment, include `-f compose.gpu.yaml` in these commands as well. After changing the model, device, image, or ports, run the appropriate `up -d` command again to recreate the affected containers. The old generated `docker-compose.yml`, if present, is not used by these commands.

## References

* [OpenHands documentation](https://docs.all-hands.dev/)
* [OpenVINO model collection](https://huggingface.co/OpenVINO)
* [OpenVINO Model Server chat API](../../docs/model_server_rest_api_chat.md)
