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
```

### Step 2: Create the model cache directory

```bash
mkdir -p "$MODEL_CACHE_DIR"
```

> **Note:** OVMS runs as a non-root user inside the container. The mounted model cache directory must be writable by the OVMS container user. If OVMS fails during startup with permission errors when creating model directories, verify permissions on the model cache directory.

### Step 3: Deploy OVMS

```bash
# Create the Docker network
docker network create ovms-net 2>/dev/null || true

# Run OVMS container
docker run -d \
    --name ovms-llm \
    --network ovms-net \
    --publish 8000:8000 \
    --publish 9000:9000 \
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
    --publish 3000:3000 \
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
curl -sf http://localhost:8000/v1/config | grep AVAILABLE
```

Or check container logs:

```bash
docker logs ovms-llm
```

---

## Understanding `docker-compose.yml`

The `docker-compose.yml` file documents the service architecture. You can achieve the same result with the manual Docker commands above.

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
  - "8000:8000"  # REST API
  - "9000:9000"  # gRPC API
```

Exposes the OpenAI-compatible REST API (8000) and gRPC API (9000).

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
  - "3000:3000"  # Web UI
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

The `scripts/deploy_model_ovms.sh` script automates the manual workflow documented above. All steps it performs can be done manually.

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
export MODEL_ID LOCAL_NAME TARGET_DEVICE TOOL_PARSER MODEL_CACHE_DIR HF_TOKEN
```

These variables are consumed by docker-compose.yml via environment variable substitution.

**6. Deploys OVMS and OpenHands**

```bash
docker compose -f "$COMPOSE_FILE" up -d
```

**7. Waits for health**

Polls `http://localhost:8000/v1/config` until the model reports `AVAILABLE` (up to 3 minutes).

**8. Prints the manual equivalent**

Shows the manual Docker commands equivalent to what the script just performed.

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
- `--compose-file FILE`: Path to docker-compose.yml
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
  --tool_parser hermes3
```

Enable TRACE logging for detailed diagnostics:

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

Enable the same logging level by modifying the OVMS command in `docker-compose.yml`. Add the `--log_level` argument:

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
  - --log_level TRACE
```

### Restarting After Configuration Changes

Restart the OVMS container for configuration changes to take effect:

```bash
docker compose restart ovms-llm
```

### Viewing TRACE Logs

Follow the logs after restarting:

```bash
docker logs -f ovms-llm
```

Observe detailed OVMS logs while reproducing an issue.
