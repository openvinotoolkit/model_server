# OpenHands Integration with OpenVINO Model Server {#ovms_demos_integration_with_openhands}

## Description

[OpenHands](https://github.com/All-Hands-AI/OpenHands) is an open-source software engineering agent that automates coding tasks through iterative LLM inference, tool execution, and runtime sandbox environments. Unlike simple chat interfaces, OpenHands maintains long-running conversations, creates code execution sandboxes, and performs multi-step problem solving.

This demo integrates OpenHands with [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) using OVMS's OpenAI-compatible REST API. It demonstrates how to deploy OVMS as a backend for OpenHands, enabling agent workflows on local hardware with OpenVINO-optimized models.

**Documentation-first approach:** This README is the authoritative source of truth. The `docker-compose.yml` and `scripts/deploy_model_ovms.sh` files are provided as optional convenience artifacts. You can complete the entire setup by following this documentation alone.

## Architecture

The integration uses standard HTTP APIs between independent components:

```text
User
  |
  v
OpenHands Container
  | (creates runtime sandbox containers for code execution)
  |
  | OpenAI-compatible HTTP requests
  | POST /v3/chat/completions
  | LLM_BASE_URL=http://ovms-llm:8000/v3
  | LLM_MODEL=openai/<served-model-name>
  v
OpenVINO Model Server
  | (MediaPipe LLM graph + tool parser)
  v
OpenVINO model
```

### Components

1. **OpenHands** — Agent application that:
   - Maintains conversation state and agent task context
   - Creates isolated Docker containers for code execution
   - Requires an OpenAI-compatible LLM endpoint for inference
   - Needs models with sufficient context capacity and coding capability

2. **OpenVINO Model Server** — Inference backend that:
   - Serves generative models through OpenAI-compatible `/v3` REST API
   - Handles model retrieval, OpenVINO conversion, and graph generation
   - Applies model-specific tool parsers for structured output
   - Runs on CPU or GPU with OpenVINO optimization

3. **OpenVINO Model** — The LLM that:
   - Must support instruction-following and code generation
   - Requires tool parser configuration for agent function calling
   - Benefits from larger context windows for complex tasks
   - Is stored externally from the Git repository

### Request Flow

1. User creates an agent task in the OpenHands web UI
2. OpenHands sends `POST /v3/chat/completions` requests to OVMS
3. OVMS routes requests through the MediaPipe LLM graph
4. OpenVINO inference engine processes the model
5. OVMS returns the completion (possibly with tool calls)
6. OpenHands parses the response and continues the agent loop

### Why OpenHands Requires Additional Configuration

Unlike simple chat UIs, OpenHands has specific requirements:

- **Model prefix:** OpenHands expects `openai/<model-name>` format in `LLM_MODEL`
- **API key placeholder:** A non-empty `LLM_API_KEY` is required even though OVMS doesn't authenticate
- **Stable networking:** Container-to-container communication on a shared Docker network
- **Docker socket access:** OpenHands creates runtime sandbox containers for code execution
- **Resource limits:** Sandbox memory limits prevent runaway agent processes

---

## Prerequisites

### System Requirements

- **Host architecture:** x86_64
- **Operating system:** Linux (Docker-based deployment)
- **Docker Engine:** Installed and running
- **Docker Compose:** Plugin v2 or standalone
- **Memory:** Minimum 8GB RAM; 16GB+ recommended for agent workflows
- **Hugging Face account:** For model access (gated models may require token)

### Network and Port Usage

The deployment uses the following ports:

| Port | Component          | Purpose                          |
|------|---------------------|----------------------------------|
| 8000 | OVMS                | OpenAI-compatible REST API      |
| 9000 | OVMS                | gRPC API (not used in this demo) |
| 3000 | OpenHands           | Web UI                           |

Ensure these ports are available on your host.

### Model Storage Location

Models, OpenVINO IR files, and generated graphs are stored externally to the Git repository. The recommended workspace is:

```text
${HOME}/ovms-openhands/
└── models/
    └── <model-id>/
        ├── openvino_model.xml
        ├── openvino_model.bin
        └── graph.pbtxt
```

This keeps the repository lightweight and allows model reuse across OVMS deployments.

---

## 4. Preparing the Model

### Choosing a Compatible Model

OpenHands requires models with:

- **Instruction-following capability:** The model must understand and execute complex prompts
- **Coding proficiency:** For software engineering agent tasks
- **Sufficient context window:** At least 4096 tokens; 8192+ recommended for complex tasks
- **Tool calling support:** For structured function output in agent workflows

Example model families that work well:

| Model Family | Tool Parser | Notes                                  |
|--------------|-------------|----------------------------------------|
| Qwen 2.5     | `qwen`      | Strong coding performance, various sizes |
| Llama 3      | `hermes3`   | Good general instruction following   |
| Mistral      | `hermes3`   | Efficient inference                  |

### Why Tool Parser Selection Matters

Tool parsers enable structured output for function calling. When OpenHands needs to execute a tool (e.g., running code, reading files), it expects the LLM to return structured JSON that specifies the tool name and arguments. The tool parser converts model outputs into this format.

- **Without tool parser:** The model may hallucinate tool calls or return unstructured text
- **With correct tool parser:** Reliable tool extraction for agent behavior

### OVMS `--source_model` Workflow

OVMS provides native model retrieval and preparation through the `--source_model` parameter:

```bash
docker run --rm -v ${HOME}/ovms-openhands/models:/models \
    openvino/model_server:latest \
    --source_model Qwen/Qwen2.5-0.5B-Instruct \
    --model_repository_path /models \
    --model_name qwen2.5-0.5b-instruct \
    --task text_generation \
    --target_device CPU
```

This command:

1. Downloads the model from Hugging Face
2. Converts to OpenVINO IR format if needed
3. Generates the MediaPipe LLM graph
4. Stores artifacts in the specified model repository

### External Model Workspace Layout

After running the `--source_model` workflow, the model directory contains:

```text
${HOME}/ovms-openhands/models/
└── qwen2.5-0.5b-instruct/
    ├── openvino_model.xml       # OpenVINO model structure
    ├── openvino_model.bin       # Model weights
    └── graph.pbtxt              # MediaPipe LLM graph configuration
```

### Using Helper Scripts (Optional)

The `scripts/deploy_model_ovms.sh` helper automates model preparation. Conceptually, it performs these steps:

1. **Validates prerequisites** — Checks Docker, docker compose, and HF_TOKEN
2. **Normalizes the model name** — Converts `Qwen/Qwen2.5-0.5B-Instruct` to `qwen2.5-0.5b-instruct`
3. **Resolves the tool parser** — Maps model family to appropriate parser (e.g., Qwen → `qwen`)
4. **Creates the model cache directory** — Ensures `${HOME}/ovms-openhands/models` exists
5. **Exports runtime configuration** — Sets environment variables for docker-compose.yml
6. **Deploys OVMS and OpenHands** — Launches containers via Docker Compose
7. **Waits for health** — Polls OVMS until the model reports `AVAILABLE`

You can achieve the same result manually by setting environment variables and running Docker commands directly, as documented in the following sections.

---

## 5. Deployment Workflow

This demo provides two equivalent paths to deploy OVMS and OpenHands. Both achieve the same result—choose based on your preference.

### Path A: Using Helper Artifacts (Convenience)

This approach uses the provided `docker-compose.yml` and `scripts/deploy_model_ovms.sh` to automate repetitive tasks.

**Prerequisites:**
- Docker Engine and Docker Compose installed
- Hugging Face token in `HF_TOKEN` environment variable (for gated models)
- 8GB+ RAM available

**Steps:**

1. **Set your Hugging Face token** (required for gated models like Llama, Mistral):
   ```bash
   export HF_TOKEN="your_token_here"
   ```

2. **Run the deployment script** with your chosen model:
   ```bash
   cd /path/to/model_server/demos/integration_with_OpenHands
   ./scripts/deploy_model_ovms.sh Qwen/Qwen2.5-0.5B-Instruct
   ```

   The script will:
   - Validate Docker and docker compose availability
   - Normalize the model name (e.g., `Qwen2.5-0.5B-Instruct` → `qwen2.5-0.5b-instruct`)
   - Resolve the appropriate tool parser (Qwen models → `qwen`)
   - Create the model cache directory at `${HOME}/ovms-openhands/models`
   - Export runtime environment variables for docker-compose.yml
   - Launch OVMS and OpenHands containers
   - Wait for OVMS to report the model as `AVAILABLE`

3. **Verify the deployment** (see Section 7)

**Optional script parameters:**
```bash
# Specify device, parser, or cache directory
./scripts/deploy_model_ovms.sh Qwen/Qwen2.5-0.5B-Instruct \
    --device CPU \
    --parser qwen \
    --cache-dir ~/custom-models

# Skip health check for faster feedback
./scripts/deploy_model_ovms.sh Qwen/Qwen2.5-0.5B-Instruct --skip-wait
```

### Path B: Manual Deployment (Documentation-Driven)

This approach uses Docker commands directly. Understanding this workflow helps you debug issues and customize the deployment.

**Note:** The commands below assume you are in the demo root directory (`model_server/demos/integration_with_OpenHands/`).

**Step 1: Set environment variables**

```bash
# Model configuration
export MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
export LOCAL_NAME="qwen2.5-0.5b-instruct"
export TARGET_DEVICE="CPU"
export TOOL_PARSER="qwen"
export MODEL_CACHE_DIR="${HOME}/ovms-openhands/models"
export HF_TOKEN="${HF_TOKEN:-}"
```

**Step 2: Create the model cache directory**

```bash
mkdir -p "$MODEL_CACHE_DIR"
```

**Step 3: Deploy OVMS**

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

This command:
- Downloads the model from Hugging Face (if not cached)
- Converts to OpenVINO IR format (if needed)
- Generates the MediaPipe LLM graph
- Starts the OVMS server with the OpenAI-compatible REST API

**Step 4: Deploy OpenHands**

```bash
# Run OpenHands container
docker run -d \
    --name openhands \
    --network ovms-net \
    --publish 3000:3000 \
    --extra-hosts host.docker.internal:host-gateway \
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

The `--extra-hosts` mapping allows the OpenHands container to reach host services if needed. It is optional for basic OVMS communication.

**Step 5: Wait for OVMS to be ready**

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

## 6. Understanding `docker-compose.yml` (Reference)

The `docker-compose.yml` file is a **reference configuration** that documents the service architecture. You can achieve the same result with manual Docker commands (see Path B above).

### Service: ovms-llm

```yaml
ovms-llm:
  image: openvino/model_server:latest
  container_name: ovms-llm
```

Uses the published OVMS image. The `container_name` provides a stable hostname for OpenHands to reach OVMS.

**Device mapping:**
```yaml
devices:
  - /dev/dri:/dev/dri
```

Provides GPU device access for hardware acceleration. For CPU-only deployments, this can be removed.

**Port publishing:**
```yaml
ports:
  - "8000:8000"  # REST API
  - "9000:9000"  # gRPC API
```

Exposes the OpenAI-compatible REST API (8000) and gRPC API (9000). This demo uses the REST API.

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

Passes the Hugging Face token for gated models. Empty by default.

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

Configures OVMS to:
- Use `/models` as the model repository
- Download the model from Hugging Face (`MODEL_ID`)
- Serve it under the local name (`LOCAL_NAME`)
- Use the text-generation pipeline
- Run on the specified device (`CPU` or `GPU`)
- Enable tool parsing for structured output

### Service: openhands

```yaml
openhands:
  image: ghcr.io/all-hands-ai/openhands:latest
  container_name: openhands
  depends_on:
    - ovms-llm
```

Uses the published OpenHands image. `depends_on` ensures OVMS starts first (though it does not wait for health).

**Port publishing:**
```yaml
ports:
  - "3000:3000"  # Web UI
```

Exposes the OpenHands web interface.

**Volume mounts:**
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
  - ./.openhands:/.openhands
```

- Docker socket: Allows OpenHands to create runtime sandbox containers for code execution
- `.openhands` directory: Persists OpenHands settings locally

**Environment variables:**
```yaml
environment:
  LLM_BASE_URL: http://ovms-llm:8000/v3
  LLM_MODEL: openai/${LOCAL_NAME}
  LLM_API_KEY: unused
```

Points OpenHands to the OVMS OpenAI-compatible endpoint. Note the `openai/` prefix required by OpenHands.

**Extra hosts:**
```yaml
extra_hosts:
  - host.docker.internal:host-gateway
```

Allows the OpenHands container to reach the host machine via `host.docker.internal`. This is optional for basic OVMS communication but may be needed for certain agent workflows that access host services.

**Generation parameters:**
```yaml
LLM_TEMPERATURE: "0.0"
LLM_MAX_OUTPUT_TOKENS: "500"
LLM_MAX_INPUT_TOKENS: "4096"
LLM_TIMEOUT: "120000"
```

Configures generation behavior:
- `temperature=0.0` for deterministic responses
- Conservative output limits to prevent runaway loops
- Increased timeout to accommodate CPU inference latency

**Sandbox limits:**
```yaml
SANDBOX_DOCKER_ARGS: --memory=1536m --memory-swap=1536m
```

Limits memory for OpenHands runtime sandboxes (agent code execution containers). Adjust based on available host RAM.

### Network

```yaml
networks:
  ovms-net:
    name: ovms-net
    driver: bridge
```

Creates a shared Docker network for container-to-container communication. OpenHands reaches OVMS via `http://ovms-llm:8000`.

---

## 7. Understanding `deploy_model_ovms.sh` (Reference)

The `scripts/deploy_model_ovms.sh` script is a **convenience helper** that automates the manual workflow documented in Path B. It is optional—all steps it performs can be done manually.

### What the Script Does

**1. Validates prerequisites**
- Checks for Docker and docker compose availability
- Warns if `HF_TOKEN` is not set for gated models
- Validates the target device (`CPU` or `GPU`)

**2. Normalizes the model name**
```bash
# "Qwen/Qwen2.5-0.5B-Instruct" → "qwen2.5-0.5b-instruct"
basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
```

**3. Resolves the tool parser**
```bash
# Maps model family to parser: Qwen → qwen, Llama3/Mistral → hermes3
```

**4. Creates the model cache directory**
```bash
mkdir -p "$MODEL_CACHE_DIR"  # Defaults to ${HOME}/ovms-openhands/models
```

**5. Exports runtime configuration**
```bash
export MODEL_ID
export LOCAL_NAME
export TARGET_DEVICE
export TOOL_PARSER
export MODEL_CACHE_DIR
export HF_TOKEN
```

These variables are consumed by docker-compose.yml via environment variable substitution.

**6. Deploys OVMS and OpenHands**
```bash
docker compose -f "$COMPOSE_FILE" up -d
```

**7. Waits for health**
Polls `http://localhost:8000/v1/config` until the model reports `AVAILABLE` (up to 3 minutes).

**8. Prints the manual equivalent**
Shows the manual Docker commands equivalent to what the script just performed, for transparency and learning.

### Script Usage

```bash
./scripts/deploy_model_ovms.sh <model_id> [OPTIONS]
```

**Arguments:**
- `model_id`: Hugging Face model ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)

**Options:**
- `--device DEVICE`: Target device (`CPU` or `GPU`, default: `CPU`)
- `--parser PARSER`: Tool parser (`hermes3`, `qwen`, or `none`, default: auto-resolved)
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

## 8. Verifying the Deployment

After deploying OVMS and OpenHands, verify the integration in two stages: first OVMS directly, then OpenHands.

### Stage 1: Verify OVMS Directly

**Check OVMS health:**

```bash
curl -s http://localhost:8000/v1/config | jq .
```

The response should include `"model_status": "AVAILABLE"` indicating the model is loaded and ready. The exact structure depends on the OVMS version, but `AVAILABLE` confirms successful deployment.

**Test a simple completion request:**

```bash
curl -X POST http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }'
```

A successful response follows the OpenAI chat completion format with `choices` containing assistant messages. The exact content varies by model.

If this succeeds, OVMS is correctly serving the model. Proceed to Stage 2.

**If OVMS fails to respond:**
- Check container logs: `docker logs ovms-llm`
- Verify the model downloaded successfully: `ls -la "$MODEL_CACHE_DIR"`
- Check if `HF_TOKEN` is needed for the model

### Stage 2: Verify OpenHands Integration

**1. Open the OpenHands web UI:**

Navigate to `http://localhost:3000` in your browser.

**2. Create a simple agent task:**

Enter a straightforward coding request, such as:

```
Create a Python function that calculates the factorial of a number.
```

**3. Observe the agent behavior:**

- OpenHands should create a runtime sandbox container
- The agent should write and test code
- The conversation should show the LLM responses from OVMS

**4. Check container logs for diagnostics:**

```bash
# OpenHands logs
docker logs openhands

# OVMS request logs
docker logs ovms-llm
```

**Successful integration indicators:**
- OpenHands UI responds to prompts
- Agent creates and uses runtime containers
- OVMS logs show incoming `/v3/chat/completions` requests
- No authentication errors (the `unused` API key is accepted)

**Common issues:**
- If OpenHands shows API errors: Verify `LLM_BASE_URL` and `LLM_MODEL` match OVMS configuration
- If responses are slow: CPU inference is expected to be slower than GPU; consider `LLM_TIMEOUT` setting
- If agent fails at tasks: The model may lack coding capability; try a larger model

---

## 9. Troubleshooting

### OVMS Container Issues

**Problem:** OVMS exits immediately after starting.

**Check:**
```bash
docker logs ovms-llm
```

**Possible causes:**
- Invalid `HF_TOKEN` for gated model → Set the correct token
- Invalid model ID → Verify the model exists on Hugging Face
- Device not available → Change `TARGET_DEVICE` to `CPU`
- Volume mount error → Ensure `MODEL_CACHE_DIR` exists and is accessible

**Problem:** OVMS starts but model status is not `AVAILABLE`.

**Check:**
```bash
curl -s http://localhost:8000/v1/config
```

**Possible causes:**
- Model still downloading → Wait longer for large models
- Out of memory → Check host RAM availability; model may be too large
- Tool parser mismatch → Verify `TOOL_PARSER` matches model family

**Problem:** Direct curl test to OVMS fails with connection refused.

**Possible causes:**
- OVMS container not running → `docker ps` to check
- Wrong port → Verify port mapping `8000:8000`
- Firewall blocking → Ensure port 8000 is accessible

### OpenHands Container Issues

**Problem:** OpenHands UI shows API errors.

**Check:**
```bash
docker logs openhands
```

**Possible causes:**
- `LLM_BASE_URL` incorrect → Should be `http://ovms-llm:8000/v3`
- `LLM_MODEL` format wrong → Should be `openai/<model-name>` with prefix
- OVMS not ready → Verify OVMS model is `AVAILABLE`

**Problem:** OpenHands fails to create runtime sandboxes.

**Check:**
```bash
docker logs openhands | grep -i sandbox
```

**Possible causes:**
- Docker socket not mounted → Verify `/var/run/docker.sock` volume mount
- Permission denied → May need to adjust Docker socket permissions
- Memory limit too low → Increase `SANDBOX_DOCKER_ARGS` memory

### Performance Issues

**Problem:** Responses are very slow.

**Considerations:**
- CPU inference is inherently slower than GPU
- First-token latency is higher for CPU-optimized models
- Model size affects speed—smaller models are faster
- Check host resource usage: `docker stats`

**Problem:** Agent tasks fail or produce poor results.

**Considerations:**
- Model may lack coding capability → Try a model optimized for code
- Context window too small → Increase `LLM_MAX_INPUT_TOKENS`
- Output limit too low → Increase `LLM_MAX_OUTPUT_TOKENS`
- Temperature too low → Try `0.1` or `0.2` for more variety

### Network Issues

**Problem:** Containers cannot communicate.

**Check:**
```bash
docker network inspect ovms-net
```

**Possible causes:**
- Containers not on same network → Verify both use `ovms-net`
- Container name wrong → OpenHands expects `ovms-llm` hostname

### Getting Help

For additional help, consult:
- OpenHands documentation: https://docs.all-hands.dev/
- OVMS documentation: https://github.com/openvinotoolkit/model_server
- This demo's `docker-compose.yml` and `scripts/deploy_model_ovms.sh` for reference

---

## 10. Screenshots

The following screenshots illustrate the integration. *(To be added during final validation)*

**Figure 1: OpenHands web UI**
- Shows the web interface at `http://localhost:3000`
- Example agent task input

**Figure 2: OVMS health check**
- Terminal output showing model status `AVAILABLE`
- Successful `/v1/config` response

**Figure 3: Successful agent task**
- OpenHands conversation with code execution
- Agent response and completed task

---

## 11. References

- **OpenHands Project:** https://github.com/All-Hands-AI/OpenHands
- **OpenHands Documentation:** https://docs.all-hands.dev/
- **OpenVINO Model Server:** https://github.com/openvinotoolkit/model_server
- **OVMS Documentation:** https://github.com/openvinotoolkit/model_server/tree/main/docs
- **Hugging Face Models:** https://huggingface.co/models
- **OpenAI API Specification:** https://platform.openai.com/docs/api-reference

### Related OVMS Demos

- **integration_with_OpenWebUI:** Similar integration pattern with a general model interface
- **llm_standalone_flow:** Standalone LLM deployment with OVMS

### Model Documentation

- **Qwen Models:** https://huggingface.co/Qwen
- **Llama Models:** https://huggingface.co/meta-llama
- **Mistral Models:** https://huggingface.co/mistralai
