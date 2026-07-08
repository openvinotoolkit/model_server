# Advanced Deployment Guide

This document contains detailed deployment and implementation reference material for the OpenHands + OVMS integration. For the Quick Start guide, see [README.md](README.md).

---

## Request Flow

1. User creates an agent task in the OpenHands web UI
2. OpenHands sends `POST /v3/chat/completions` requests to OVMS
3. OVMS routes requests through the MediaPipe LLM graph
4. OpenVINO inference engine processes the model
5. OVMS returns the completion (possibly with tool calls)
6. OpenHands parses the response and continues the agent loop

---

## Why OpenHands Requires Additional Configuration

Unlike simple chat UIs, OpenHands has specific requirements:

- **Model prefix:** OpenHands expects `openai/<model-name>` format in `LLM_MODEL`
- **API key placeholder:** A non-empty `LLM_API_KEY` is required even though OVMS doesn't authenticate
- **Stable networking:** Container-to-container communication on a shared Docker network
- **Docker socket access:** OpenHands creates runtime sandbox containers for code execution
- **Resource limits:** Sandbox memory limits prevent runaway agent processes

---

## OVMS `--source_model` Workflow

OVMS provides native model retrieval and preparation through the `--source_model` parameter:

```bash
docker run --rm -v ${HOME}/ovms-openhands/models:/models \
    openvino/model_server:latest \
    --source_model OpenVINO/Qwen3-8b-int8-ov \
    --model_repository_path /models \
    --model_name qwen3-8b-int8-ov \
    --task text_generation \
    --target_device CPU
```

This command downloads the model from Hugging Face, converts to OpenVINO IR format if needed, generates the MediaPipe LLM graph, and stores artifacts in the specified model repository.

---

## Model Workspace Layout

After running the `--source_model` workflow, the model directory contains:

```text
${HOME}/ovms-openhands/models/
└── qwen3-8b-int8-ov/
    ├── openvino_model.xml       # OpenVINO model structure
    ├── openvino_model.bin       # Model weights
    ├── graph.pbtxt              # MediaPipe LLM graph configuration
    └── .......
```

This external storage keeps the Git repository lightweight and allows model reuse across OVMS deployments.

---

## Manual Deployment Workflow

You can deploy using Docker commands directly without the helper scripts. This approach is useful for debugging and customization.

**Repository not required:** These commands can be executed from any directory on a Linux system with Docker installed. The OpenHands state directory (`.openhands`) will be created relative to your current working directory.

### Step 1: Set environment variables

```bash
# Model configuration
export MODEL_ID="OpenVINO/Qwen3-8b-int8-ov"
export LOCAL_NAME="qwen3-8b-int8-ov"
export TARGET_DEVICE="CPU"
export TOOL_PARSER="hermes3"
export MODEL_CACHE_DIR="${HOME}/ovms-openhands/models"
export HF_TOKEN="${HF_TOKEN:-}"

# Published ports (optional - defaults shown)
export OVMS_REST_PORT="${OVMS_REST_PORT:-8000}"
export OVMS_GRPC_PORT="${OVMS_GRPC_PORT:-9000}"
export OPENHANDS_PORT="${OPENHANDS_PORT:-3000}"

# Proxy configuration (optional - forward to containers if set)
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export HTTP_PROXY="${HTTP_PROXY:-}"
export HTTPS_PROXY="${HTTPS_PROXY:-}"
export no_proxy="${no_proxy:-}"
export NO_PROXY="${NO_PROXY:-}"
```

### Step 2: Create the model cache directory

```bash
mkdir -p "$MODEL_CACHE_DIR"
```

> **Note:** OVMS runs as a non-root user inside the container. The mounted model cache directory must be writable by the OVMS container user.
>
> On some WSL2/Docker environments, the mounted directory permissions may prevent OVMS from creating model directories. If OVMS fails during startup with permission errors, you may see an error like:
>
> ```
> Libgit2 clone error: failed to make directory '/models/...': Permission denied
> ```
>
> Verify the directory permissions:
>
> ```bash
> ls -ld "$MODEL_CACHE_DIR"
> ```
>
> Fix by making the directory writable by all users:
>
> ```bash
> chmod a+rwx "$MODEL_CACHE_DIR"
> ```
>
> Then redeploy the OVMS container.

### Step 3: Deploy OVMS

```bash
# Create the Docker network
docker network create ovms-net 2>/dev/null || true

# Run OVMS container
docker run -d \
    --name ovms-llm \
    --network ovms-net \
    --publish ${OVMS_REST_PORT}:8000 \
    --publish ${OVMS_GRPC_PORT}:9000 \
    --device /dev/dri:/dev/dri \
    --volume "$MODEL_CACHE_DIR:/models:rw" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --restart unless-stopped \
    openvino/model_server:latest \
    --model_repository_path /models \
    --source_model "$MODEL_ID" \
    --model_name "$LOCAL_NAME" \
    --task text_generation \
    --target_device "$TARGET_DEVICE" \
    --port 9000 \
    --rest_port 8000 \
    --tool_parser "$TOOL_PARSER"
```

This command downloads the model from Hugging Face (if not cached), converts to OpenVINO IR format (if needed), generates the MediaPipe LLM graph, and starts the OVMS server with the OpenAI-compatible REST API.

### Step 4: Deploy OpenHands

```bash
# Run OpenHands container
docker run -d \
    --name openhands \
    --network ovms-net \
    --publish ${OPENHANDS_PORT}:3000 \
    --add-host host.docker.internal:host-gateway \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume "$(pwd)/.openhands:/.openhands" \
    --env LLM_BASE_URL="http://ovms-llm:8000/v3" \
    --env LLM_MODEL="openai/${LOCAL_NAME}" \
    --env LLM_API_KEY="unused" \
    --env LLM_TEMPERATURE="0.0" \
    --env LLM_MAX_OUTPUT_TOKENS="500" \
    --env LLM_MAX_INPUT_TOKENS="4096" \
    --env LLM_TIMEOUT="120000" \
    --env SANDBOX_DOCKER_ARGS="--memory=1536m --memory-swap=1536m" \
    --restart unless-stopped \
    ghcr.io/all-hands-ai/openhands:latest
```

The `--add-host` mapping allows the OpenHands container to reach host services if needed. It is optional for basic OVMS communication.

### Step 5: Wait for OVMS to be ready

OVMS needs time to download and initialize the model. Check the status:

```bash
# Poll until model is AVAILABLE
curl -sf http://localhost:${OVMS_REST_PORT}/v1/config | grep AVAILABLE
```

Or check container logs:

```bash
docker logs ovms-llm
```

---

## Compose Generation Workflow

The deployment uses three files with distinct responsibilities:

### docker-compose.template.yml

Committed to Git, this file serves as the canonical template for deployment structure. It documents the service architecture and contains environment variable placeholders that the deployment script substitutes with runtime values.

The template uses placeholders like `${MODEL_ID}`, `${LOCAL_NAME}`, `${TARGET_DEVICE}`, and others. The deployment script uses `envsubst` to replace these placeholders with actual values when generating the runtime compose file.

Routine deployment customizations should be made through deployment script options, environment variables, or by editing the generated `docker-compose.yml`. Modify the template only when changing the deployment structure itself.

### docker-compose.yml

Generated from the template by the deployment script, this file represents the active deployment. It is gitignored and may be edited locally for customization.

Standard Docker Compose commands operate on this file:
```bash
docker compose up -d
docker compose down
docker compose restart
docker compose restart openhands
docker compose restart ovms-llm
docker compose logs -f ovms-llm
```

Local modifications to the generated compose are preserved as long as the deployment configuration remains unchanged. See Deployment Lifecycle below.

### .ovms-deployment

Generated by the deployment script, this file stores deployment metadata (deployment fingerprint). The deployment script compares this metadata against the requested configuration to determine whether the existing `docker-compose.yml` can be reused or must be regenerated.

The deployment fingerprint includes all parameters that affect compose generation:

- Model identifier and local name
- Target device
- Tool and reasoning parsers
- OVMS image variant
- GPU device mapping
- Model cache directory
- Published ports
- Proxy configuration
- Metadata version and generation timestamp

This file is for internal use by the deployment script and should not be edited manually.

### Deployment Lifecycle

The deployment script uses the deployment fingerprint to determine whether to preserve or regenerate the compose file.

**First deployment:** If no generated compose exists, the script generates `docker-compose.yml` from the template, creates `.ovms-deployment`, and deploys with Docker Compose.

**Redeploying an identical deployment:** If the stored deployment fingerprint matches the requested deployment (all parameters identical), the script preserves the existing `docker-compose.yml` and all local user modifications, then deploys using the existing compose. This allows users to make local compose customizations without having those changes discarded.

**Deploying a different configuration:** If any deployment fingerprint parameter changes (model, device, parser, ports, proxy settings, cache directory, etc.), the script stops the existing deployment, removes the generated compose, regenerates it from the template, regenerates deployment metadata, and deploys the new configuration. This resets local compose modifications because the deployment itself has changed.

---

## Template Architecture

The `docker-compose.template.yml` file documents the service architecture. See the Compose Generation Workflow section above for details on how the deployment script generates the runtime compose file from this template.

### Service: ovms-llm

```yaml
ovms-llm:
  image: openvino/model_server:latest
  container_name: ovms-llm
```

The `container_name` provides a stable hostname for OpenHands to reach OVMS.

**Device mapping:**
```yaml
devices:
  - /dev/dri:/dev/dri
```

Provides GPU device access. For CPU-only deployments, this can be removed.

**Port publishing:**
```yaml
ports:
  - "${OVMS_REST_PORT}:8000"  # REST API (default: 8000)
  - "${OVMS_GRPC_PORT}:9000"  # gRPC API (default: 9000)
```

Exposes the OpenAI-compatible REST API and gRPC API. The published host ports are configurable through environment variables (`OVMS_REST_PORT`, `OVMS_GRPC_PORT`), with defaults of 8000 and 9000 respectively.

**Volume mount:**
```yaml
volumes:
  - ${MODEL_CACHE_DIR:-./docker/models}:/models:rw
```

Mounts the model cache directory where OVMS materializes models via `--source_model`. The script sets `MODEL_CACHE_DIR` to `${HOME}/ovms-openhands/models` by default; the compose file fallback is `./docker/models` if the variable is unset.

**Environment:**
```yaml
environment:
  HF_TOKEN: ${HF_TOKEN:-}
```

Passes the Hugging Face token for gated models.

**Command:**
```yaml
command:
  - --model_repository_path /models
  - --source_model ${MODEL_ID}
  - --model_name ${LOCAL_NAME}
  - --task text_generation
  - --target_device ${TARGET_DEVICE}
  - --port "9000"
  - --rest_port "8000"
  - --tool_parser ${TOOL_PARSER}
```

Configures OVMS to use `/models` as the repository, download from Hugging Face, serve under the local name, use the text-generation pipeline, run on the specified device, and enable tool parsing.

### Service: openhands

```yaml
openhands:
  image: ghcr.io/all-hands-ai/openhands:latest
  container_name: openhands
  depends_on:
    - ovms-llm
```

`depends_on` ensures OVMS starts first (though it does not wait for health).

**Port publishing:**
```yaml
ports:
  - "${OPENHANDS_PORT}:3000"  # Web UI (default: 3000)
```

**Volume mounts:**
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
  - ./.openhands:/.openhands
```

Docker socket allows OpenHands to create runtime sandbox containers. The `.openhands` directory persists settings locally.

**Environment variables:**
```yaml
environment:
  LLM_BASE_URL: http://ovms-llm:8000/v3
  LLM_MODEL: openai/${LOCAL_NAME}
  LLM_API_KEY: unused
```

Points OpenHands to the OVMS endpoint. Note the `openai/` prefix required by OpenHands.

**Extra hosts:**
```yaml
extra_hosts:
  - host.docker.internal:host-gateway
```

Allows the OpenHands container to reach the host machine. Optional for basic OVMS communication but may be needed for certain agent workflows.

**Generation parameters:**
```yaml
LLM_TEMPERATURE: "0.0"
LLM_MAX_OUTPUT_TOKENS: "500"
LLM_MAX_INPUT_TOKENS: "4096"
LLM_TIMEOUT: "120000"
```

Temperature set to 0.0 for deterministic responses. Conservative output limits prevent runaway loops. Increased timeout accommodates CPU inference latency.

**Sandbox limits:**
```yaml
SANDBOX_DOCKER_ARGS: --memory=1536m --memory-swap=1536m
```

Limits memory for OpenHands runtime sandboxes. Adjust based on available host RAM.

### Network

```yaml
networks:
  ovms-net:
    name: ovms-net
    driver: bridge
```

Creates a shared Docker network for container-to-container communication. OpenHands reaches OVMS via `http://ovms-llm:8000`.

---

## Understanding `deploy_model_ovms.sh`

The `scripts/deploy_model_ovms.sh` script automates the deployment workflow using the template-based compose generation system. All steps it performs can be done manually using the documented Docker commands.

### What the Script Does

**1. Validates prerequisites**

Checks for Docker and docker compose availability, warns if `HF_TOKEN` is not set for gated models, and validates the target device (`CPU` or `GPU`).

**2. Normalizes the model name**

```bash
# "OpenVINO/Qwen3-8b-int8-ov" → "qwen3-8b-int8-ov"
basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
```

**3. Resolves the tool parser**

Maps model family to parser (e.g., Qwen → `hermes3`, Llama3 → `llama3`, Mistral → `mistral`).

**4. Creates the model cache directory**

```bash
mkdir -p "$MODEL_CACHE_DIR"  # Defaults to ${HOME}/ovms-openhands/models
```

**5. Exports runtime configuration**

```bash
export MODEL_ID LOCAL_NAME TARGET_DEVICE TOOL_PARSER REASONING_PARSER MODEL_CACHE_DIR HF_TOKEN
export OVMS_IMAGE GPU_DEVICE WSL_LIBS
export OVMS_REST_PORT OVMS_GRPC_PORT OPENHANDS_PORT
export http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
```

These variables are consumed by the generated docker-compose.yml via environment variable substitution.

**6. Deploys OVMS and OpenHands**

The script determines the deployment action based on existing state:

- **No existing compose:** Generates `docker-compose.yml` from the template, writes `.ovms-deployment` metadata, and deploys.
- **Existing compose with matching fingerprint:** Preserves the existing compose and user modifications, updates the metadata timestamp, and deploys.
- **Existing compose with different fingerprint:** Stops the deployment, removes the old compose, regenerates from the template, writes new metadata, and deploys.

Deployment proceeds with:
```bash
docker compose -f "$COMPOSE_FILE" up -d
```

**7. Waits for health**

Polls `http://localhost:${OVMS_REST_PORT}/v1/config` until the model reports `AVAILABLE` (up to 5 minutes).

**8. Prints the manual equivalent**

Shows the manual Docker commands equivalent to what the script just performed.

### Deployment Fingerprint

The deployment fingerprint stored in `.ovms-deployment` includes all parameters that affect compose generation:

- `MODEL_ID` — Hugging Face model identifier
- `TARGET_DEVICE` — CPU or GPU
- `LOCAL_NAME` — Normalized model name
- `TOOL_PARSER` — Tool parser for structured output
- `REASONING_PARSER` — Reasoning parser for chain-of-thought
- `OVMS_IMAGE` — CPU or GPU variant
- `GPU_DEVICE` — Device mapping for GPU passthrough
- `MODEL_CACHE_DIR` — Model cache directory
- `OVMS_REST_PORT`, `OVMS_GRPC_PORT`, `OPENHANDS_PORT` — Published ports
- `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `no_proxy`, `NO_PROXY` — Proxy configuration

When any of these parameters change between deployments, the script regenerates the compose file. Otherwise, it preserves local modifications.

### Script Usage

```bash
./scripts/deploy_model_ovms.sh <model_id> [OPTIONS]
```

**Arguments:**
- `model_id`: Hugging Face model ID (e.g., `OpenVINO/Qwen3-8b-int8-ov`)

**Options:**
- `--device DEVICE`: Target device (`CPU` or `GPU`, default: `CPU`)
- `--parser PARSER`: Override the automatically resolved tool parser
- `--cache-dir DIR`: Model cache directory (default: `${HOME}/ovms-openhands/models`)
- `--compose-file FILE`: Path to docker-compose.yml (default: generated from template)
- `--skip-wait`: Skip health check and return immediately after deploy

**Environment variable overrides:**
- `HF_TOKEN`: Hugging Face token for gated models
- `LOCAL_NAME`: Override the auto-normalized model name
- `MODEL_CACHE_DIR`: Override model cache directory
- `TARGET_DEVICE`: Override target device
- `TOOL_PARSER`: Override tool parser

---

## Debugging OVMS

### Viewing OVMS Logs

View OVMS logs from the Docker container:

```bash
docker logs ovms-llm
```

Follow logs in real time while reproducing an issue:

```bash
docker logs -f ovms-llm
```

### Running OVMS with TRACE Logging

OVMS supports configurable logging levels. The following command demonstrates enabling TRACE logging for a standalone OVMS deployment:

```bash
ovms \
  --rest_port 9001 \
  --model_repository_path ./models \
  --source_model OpenVINO/Qwen3-8b-int8-ov \
  --task text_generation \
  --target_device CPU \
  --model_name qwen3-8b-int8-ov \
  --tool_parser hermes3 \
  --log_level TRACE
```

TRACE logging provides detailed information helpful for diagnosing issues related to model loading, request processing, inference, and tool-calling behavior.

### Enabling TRACE Logging with Docker Compose

To enable TRACE logging for a compose-based deployment, edit the generated `docker-compose.yml` and modify the OVMS command to include the `--log_level` argument.

The OVMS service in the generated compose uses a shell command to conditionally add parser arguments. Append the log level argument to the command array:

```yaml
command:
  - /bin/bash
  - -c
  - |
    CMD_ARGS=(
      --model_repository_path /models
      --source_model "$${MODEL_ID}"
      --model_name "$${LOCAL_NAME}"
      --task text_generation
      --target_device "$${TARGET_DEVICE}"
      --port "9000"
      --rest_port "8000"
    )
    if [[ "$${TOOL_PARSER}" != "none" ]]; then
      CMD_ARGS+=(--tool_parser "$${TOOL_PARSER}")
    fi
    if [[ "$${REASONING_PARSER}" != "none" ]]; then
      CMD_ARGS+=(--reasoning_parser "$${REASONING_PARSER}")
    fi
    CMD_ARGS+=(--log_level TRACE)
    exec ovms "$${CMD_ARGS[@]}"
```

Append the following line before `exec ovms`:

```yaml
CMD_ARGS+=(--log_level TRACE)
```

### Applying Configuration Changes

After editing the generated compose file, apply the changes using standard Docker Compose commands:

```bash
# Restart the OVMS service only
docker compose restart ovms-llm

# Or recreate the service
docker compose up -d ovms-llm
```

### Viewing TRACE Logs

View logs after restarting:

```bash
docker logs -f ovms-llm
```

---

# GPU Acceleration

## Overview

OVMS supports GPU inference through OpenVINO. The deployment script already supports GPU deployment using the `--device GPU` flag:

```bash
./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov --device GPU
```

GPU enablement depends on the host environment rather than OVMS itself. The same OVMS container and model work on CPU or GPU—only the target device flag changes. Your host system must expose an OpenVINO-compatible Intel GPU runtime for GPU inference to function.

> **Note:** Platform-specific GPU setup procedures may change with future OpenVINO, Intel GPU runtime, or OS releases. Always refer to the latest OpenVINO and Intel GPU documentation for your platform.

## Supported Platforms

GPU inference is supported on the following platforms:

| Platform                | Status         | Notes                                              |
| ----------------------- | -------------- | -------------------------------------------------- |
| Native Linux            | ✅ Supported   | Requires Intel GPU runtime and device access      |
| WSL2 + Docker Desktop   | ✅ Supported   | Requires WSL2 GPU support and `/dev/dxg`          |
| Native Windows          | ❌ Not supported | Use WSL2 for Docker-based deployment              |


## Verifying GPU Availability

Before deploying with GPU, verify that your host system can access an Intel GPU through OpenVINO.

### GPU Runtime Installation

GPU inference requires an OpenVINO-compatible Intel GPU runtime on your host system. Installation procedures vary by platform and distribution. Refer to the official documentation:

* **OpenVINO GPU documentation:** https://docs.openvino.ai/
* **Intel GPU runtime documentation:** https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/documentation.html
* **WSL2 GPU setup:** https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

Follow the official instructions for your platform before proceeding with verification.

### 1. Verify OpenVINO Detects GPU

Install OpenVINO on your host and verify GPU availability:

```bash
python3 -c "from openvino import Core; print(Core().available_devices)"
```

Expected output:

```text
['CPU', 'GPU']
```

If only `['CPU']` is returned, the GPU runtime is not installed correctly or no compatible GPU is available.

### 2. Verify GPU Device Access

**Native Linux:** Check that GPU devices are accessible:

```bash
ls -la /dev/dri/
```

You should see `renderD*` and `card*` devices.

**WSL2:** Verify the DirectX device exists:

```bash
ls -la /dev/dxg
```

Expected output:

```text
crw-rw-rw- 1 root root 247, 0 ... /dev/dxg
```

> **Note:** Under WSL2, `/dev/dri` is **not expected** to be present. GPU access is provided through `/dev/dxg` instead. Native Linux exposes GPUs through `/dev/dri`, while WSL2 exposes GPUs through `/dev/dxg`.


If only `['CPU']` is returned, ensure Docker Desktop WSL2 integration is enabled with GPU support.

### 3. Deploy with GPU

Run the deployment script with GPU device:

```bash
./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov --device GPU
```

### 5. Verify OVMS GPU Usage

Check that OVMS detects the GPU:

```bash
docker logs ovms-llm | grep -i "available devices"
```

Expected output:

```text
Available devices for Open VINO: CPU, GPU
```

Verify the model reaches `AVAILABLE` status:

```bash
curl -s http://localhost:${OVMS_REST_PORT}/v1/config | grep AVAILABLE
```

Should return output containing `AVAILABLE`.

## Troubleshooting

### GPU Not Detected

If `python3 -c "from openvino import Core; print(Core().available_devices)"` returns only `['CPU']`:

1. Verify Intel GPU is physically present: `lspci | grep -Ei "vga|display"`
2. Refer to the GPU runtime installation documentation for your platform
3. Check OpenVINO installation: `pip3 show openvino`
4. Try restarting the WSL instance: `wsl --shutdown` (from Windows)

### `/dev/dri` Missing Under WSL2

This is expected behavior under WSL2. GPU access is provided through `/dev/dxg` instead:

```bash
ls -la /dev/dxg  # Should exist
ls -la /dev/dri  # May not exist—this is normal
```

Native Linux exposes GPUs through `/dev/dri`, while WSL2 exposes GPUs through `/dev/dxg`. The docker-compose.template.yml includes `/dev/dri` device mapping for native Linux deployments.

### Docker Cannot Access GPU

If OVMS logs show no GPU available:

1. Ensure Docker Desktop (Windows) or Docker daemon (Linux) has GPU access enabled
2. On WSL2, verify Docker Desktop WSL2 integration is enabled
3. Check container logs: `docker logs ovms-llm | grep -i gpu`
4. Verify the `--device GPU` flag is being passed to OVMS

### GPU Plugin Unavailable

If OVMS starts but falls back to CPU despite `--device GPU`:

```bash
docker logs ovms-llm | grep -i "target device"
```

Look for warnings about plugin loading. This may indicate:
- GPU runtime not available to the container
- Incompatible OpenVINO version

### OpenVINO Python Not Installed

If running verification commands produces:

```text
ModuleNotFoundError: No module named 'openvino'
```

This means the OpenVINO Python package is not installed in the active Python environment, or the correct virtual environment is not activated. Install OpenVINO or activate the appropriate environment before running the verification commands.


## Performance Verification

Verify GPU acceleration is working by comparing CPU and GPU performance with identical prompts.

**1. Run a test prompt on CPU:**

```bash
curl -X POST http://localhost:${OVMS_REST_PORT}/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-int8-ov",
    "messages": [{"role": "user", "content": "Explain recursion in programming with examples."}],
    "max_tokens": 200
  }'
```

Record the total response time.

**2. Redeploy with GPU:**

```bash
docker compose -f docker-compose.yml down
./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov --device GPU
```

**3. Run the same prompt on GPU:**

Use the same curl command and compare response times.

**4. Metrics to compare:**

- **Total response time:** Compare end-to-end request duration
- **Tokens per second:** Calculate by dividing output tokens by generation time
- **CPU utilization:** GPU inference typically shows different CPU utilization patterns

GPU acceleration is generally expected to improve throughput and reduce response time for sufficiently large inference workloads. Actual performance depends on the model, hardware, prompt length, and runtime configuration.

> **Note:** The first inference on GPU includes model compilation overhead. Treat the first request as a warm-up; subsequent requests will reflect true GPU performance.

**Example verification:**

```bash
# Time the request
time curl -X POST http://localhost:${OVMS_REST_PORT}/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-int8-ov",
    "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime."}],
    "max_tokens": 150
  }'
```

Compare the `real` time between CPU and GPU runs.
