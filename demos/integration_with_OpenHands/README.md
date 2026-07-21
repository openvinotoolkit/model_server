# OpenHands Integration with OpenVINO Model Server {#ovms_demos_integration_with_openhands}

## Description

[OpenHands](https://github.com/All-Hands-AI/OpenHands) is an open-source software engineering agent that automates coding tasks through iterative LLM inference, tool execution, and runtime sandbox environments. Unlike simple chat interfaces, OpenHands maintains long-running conversations, creates code execution sandboxes, and performs multi-step problem solving.

This demo integrates OpenHands with [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) using OVMS's OpenAI-compatible REST API. It demonstrates how to deploy OVMS as a backend for OpenHands, enabling agent workflows on local hardware with OpenVINO-optimized models.

This README covers the recommended deployment workflow. For manual Docker deployment and implementation details, see [ADVANCED_DEPLOYMENT.md](ADVANCED_DEPLOYMENT.md).

## Architecture

```
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

OpenHands maintains conversation state and creates isolated Docker containers for code execution. It requires an OpenAI-compatible LLM endpoint with models that have sufficient context capacity and coding capability.

OVMS serves generative models through an OpenAI-compatible REST API, handling model retrieval, OpenVINO conversion, and graph generation. It applies model-specific tool parsers for structured output and runs on CPU or GPU with OpenVINO optimization.

For detailed request flow and configuration requirements, see [ADVANCED_DEPLOYMENT.md](ADVANCED_DEPLOYMENT.md).

---

## Prerequisites

- **Host architecture:** x86_64
- **Operating system:** Linux (Docker-based deployment)
- **Docker Engine:** Installed and running
- **Docker Compose:** Plugin v2 or standalone
- **Memory:** Minimum 8GB RAM; 16GB+ recommended for agent workflows
- **Hugging Face account:** For model access (gated models may require token)

### Network and Port Usage

| Port | Component | Purpose                     |
|------|-----------|-----------------------------|
| 8000 | OVMS      | OpenAI-compatible REST API (default)  |
| 9000 | OVMS      | gRPC API (not used here, default)    |
| 3000 | OpenHands | Web UI (default)                      |

The default published ports are 8000 (OVMS REST), 9000 (OVMS gRPC), and 3000 (OpenHands). You can override these defaults by setting environment variables before running the deployment script:

```bash
export OVMS_REST_PORT=18000
export OVMS_GRPC_PORT=19000
export OPENHANDS_PORT=3300

./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov
```

Ensure the required ports are available on your host.

### Proxy Support

In environments requiring HTTP/HTTPS proxies, export standard proxy environment variables before running the deployment script:

```bash
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov
```

The deployment script automatically forwards these variables to the Docker containers.

---

## Preparing the Model

### Choosing a Compatible Model

OpenHands requires models with instruction-following capability, coding proficiency, sufficient context window (4096+ tokens), and tool calling support.

| Model Family | Tool Parser | Notes                                  |
|--------------|-------------|----------------------------------------|
| Qwen 3 Coder | `qwen3coder`| Strong coding performance, various sizes |
| Qwen 3       | `hermes3`   | General instruction following          |
| Llama 3      | `llama3`    | Good general instruction following   |
| Mistral      | `mistral`   | Efficient inference                  |

> **Note:** Examples use `OpenVINO/Qwen3-8b-int8-ov`. Other compatible models may also be used.

### Tool Parser Selection

Tool parsers enable structured output for function calling. When OpenHands executes a tool (running code, reading files), it expects the LLM to return structured JSON specifying the tool name and arguments. The tool parser converts model outputs into this format.

Without the correct tool parser, the model does not generate tool calls in the expected format, causing tool call extraction to fail.

For details on OVMS model retrieval and workspace layout, see [ADVANCED_DEPLOYMENT.md](ADVANCED_DEPLOYMENT.md).

---

## Deployment

This demo uses a deployment template and helper script to automate container orchestration. The deployment architecture separates the template (tracked in Git) from the generated compose file (used for runtime).

### Compose Files and Responsibilities

The repository contains three compose-related files with distinct purposes:

| File | Purpose | Managed By |
|------|---------|------------|
| `docker-compose.template.yml` | Deployment template with placeholders | Git (source of truth for deployment structure) |
| `docker-compose.yml` | Generated compose file for active deployment | Deployment script (may be edited locally) |
| `.ovms-deployment` | Deployment metadata (deployment fingerprint) | Deployment script (internal use) |

**Template (`docker-compose.template.yml`):** Committed to Git, serves as the canonical template for deployment structure. Contains environment variable placeholders that the deployment script substitutes with runtime values. Routine customizations should be made through deployment script options or by editing the generated compose file. Modify the template only when changing the deployment structure itself.

**Generated compose (`docker-compose.yml`):** Created from the template on first deployment, gitignored. Represents the active deployment and may be edited locally for customization. Standard Docker Compose commands operate on this file.

**Deployment metadata (`.ovms-deployment`):** Generated by the deployment script. Stores metadata describing the generated deployment. The deployment script compares this metadata against the requested configuration to determine whether the existing `docker-compose.yml` can be reused or must be regenerated. This deployment fingerprint is compared on subsequent runs. Not intended for manual editing.

### Deployment Lifecycle

The deployment script uses the deployment fingerprint to determine whether to preserve or regenerate the compose file:

**First deployment:** If no generated compose exists, the script generates `docker-compose.yml` from the template, creates `.ovms-deployment`, and deploys with Docker Compose.

**Redeploying an identical deployment:** If the stored deployment fingerprint matches the requested deployment, the script preserves the existing `docker-compose.yml` and all local user modifications, then deploys using the existing compose.

**Deploying a different configuration:** If the deployment fingerprint changes (model, device, parser, ports, proxy settings, cache directory, etc.), the script stops the existing deployment, removes the generated compose, regenerates it from the template, regenerates deployment metadata, and deploys the new configuration.

This design allows users to make local compose customizations (such as enabling TRACE logging) without having those changes discarded when redeploying the same configuration.

### Running the Deployment

Clone the repository and navigate to the demo directory before proceeding.

**Prerequisites:** Docker Engine, Docker Compose, 8GB+ RAM, and `HF_TOKEN` for gated models.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/openvinotoolkit/model_server.git
   cd model_server/demos/integration_with_OpenHands
   ```

2. **Set your Hugging Face token** (required for gated models like Llama, Mistral):
   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. **Run the deployment script:**
   ```bash
   ./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov
   ```

   The script validates your environment, prepares the model, generates the compose file if needed, and launches both containers.

4. **Verify the deployment** (see next section)

### Using Intel GPU (Optional)

The deployment script supports GPU inference for improved performance:

```bash
./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov --device GPU
```

Before using GPU inference, ensure your host system exposes an OpenVINO-compatible Intel GPU runtime (see [GPU Acceleration](ADVANCED_DEPLOYMENT.md#gpu-acceleration) for platform-specific setup and verification). GPU support depends on your host environment—the same OVMS deployment works on CPU or GPU based on the `--device` flag.

**Optional parameters:**
```bash
# Specify device, parser, or cache directory
./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov \
    --device CPU \
    --parser hermes3 \
    --cache-dir ~/custom-models

# Skip health check for faster feedback
./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov --skip-wait
```

### Container Management

Use Docker Compose commands and the deployment script as complementary workflows:

#### Standard Docker Compose Commands

Use standard Docker Compose commands for routine container operations and local modifications to the generated compose file:

```bash
# View status
docker compose ps

# View logs
docker compose logs
docker compose logs -f ovms-llm

# Restart all services
docker compose restart

# Restart only OpenHands
docker compose restart openhands

# Restart only OVMS
docker compose restart ovms-llm

# Stop services
docker compose stop

# Start services
docker compose start

# Remove services (preserves compose file)
docker compose down
```

These commands operate on `docker-compose.yml` in the current directory.

#### Deployment Script

Run `deploy_model_ovms.sh` when changing deployment configuration:

- Deploying a different model
- Switching between CPU and GPU
- Changing tool parser or reasoning parser
- Changing published ports
- Changing proxy configuration
- Regenerating the deployment

The script compares the requested configuration against the stored deployment metadata and regenerates the compose file only when the configuration has changed.

For manual Docker deployment and implementation details, see [ADVANCED_DEPLOYMENT.md](ADVANCED_DEPLOYMENT.md).

---

## Verifying the Deployment

Verify the integration in two stages: first OVMS directly, then OpenHands.

### Stage 1: Verify OVMS

The deployment script sets the published OVMS REST port (default: 8000). Use this port for verification.

**Check health:**
```bash
# Substitute your configured OVMS_REST_PORT if not using the default
curl -s http://localhost:${OVMS_REST_PORT:-8000}/v1/config | jq .
```

The response should include `"model_status": "AVAILABLE"`.

**Test a completion request:**
```bash
curl -X POST http://localhost:${OVMS_REST_PORT:-8000}/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-int8-ov",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }'
```

If OVMS fails to respond, check `docker logs ovms-llm`, verify the model downloaded at `$MODEL_CACHE_DIR`, and ensure `HF_TOKEN` is set if needed.

### Stage 2: Verify OpenHands

The deployment script sets the published OpenHands port (default: 3000).

1. **Open the web UI** at `http://localhost:${OPENHANDS_PORT:-3000}`
   ![OpenHands UI](<screenshots/Pasted image (2).png>)

2. **Configure the OVMS-backed model:**
   - Click **Settings** → **LLM**
   - Enable **Advanced** mode if needed
   - Set **Custom Model:** `openai/qwen3-8b-int8-ov`
   - Set **Base URL:** `http://ovms-llm:8000/v3`
   - Set **API Key:** `unused`
   - Click **Save**

   ![OpenHands LLM Configuration](screenshots/Pasted%20image.png)

3. **Create an agent task:**
   ```
   Create a Python function that calculates the factorial of a number.
   ```

4. **Verify behavior:**
   - OpenHands creates a runtime sandbox container
   - The agent writes and tests code
   - OVMS logs show incoming `/v3/chat/completions` requests

**Common issues:**
- API errors: Verify `LLM_BASE_URL` and `LLM_MODEL` match OVMS configuration
- Slow responses: CPU inference is slower than GPU; consider `LLM_TIMEOUT` setting
- Task failures: The model may lack coding capability; try a larger model

---

## Troubleshooting

### OVMS Container Issues

**OVMS exits immediately after starting**

Check `docker logs ovms-llm`. Possible causes:
- Invalid `HF_TOKEN` for gated model
- Invalid model ID
- Device not available (change `TARGET_DEVICE` to `CPU`)
- Volume mount error (ensure `MODEL_CACHE_DIR` exists)
- Permission denied on `/models` (directory must be writable by OVMS container user)

**Permission denied on `/models`**

On some WSL2/Docker configurations, OVMS may fail to start with an error like:

```
Libgit2 clone error: failed to make directory '/models/OpenVINO': Permission denied
```

This occurs when the mounted model cache directory is not writable by the OVMS container user. The issue is specific to the permissions of the host directory, not OVMS itself.

Verify the directory permissions:

```bash
ls -ld "$MODEL_CACHE_DIR"
```

Fix by making the directory writable by all users:

```bash
chmod a+rwx "$MODEL_CACHE_DIR"
```

Then redeploy:

```bash
docker compose down
./scripts/deploy_model_ovms.sh <model_id>
```

**Model status is not `AVAILABLE`**

Check `curl -s http://localhost:8000/v1/config` (default port; override with `OVMS_REST_PORT`). Possible causes:
- Model still downloading (wait longer for large models)
- Out of memory (check host RAM; model may be too large)
- Tool parser mismatch (verify `TOOL_PARSER` matches model family)

**Connection refused**

Possible causes:
- OVMS container not running (`docker ps`)
- Wrong port mapping (verify the published OVMS REST port matches your configured `OVMS_REST_PORT` setting)
- Firewall blocking the configured port

### OpenHands Container Issues

**API errors**

Check `docker logs openhands`. Possible causes:
- `LLM_BASE_URL` incorrect (should be `http://ovms-llm:8000/v3`)
- `LLM_MODEL` format wrong (should be `openai/<model-name>`)
- OVMS not ready (verify model is `AVAILABLE`)

**Fails to create runtime sandboxes**

Check `docker logs openhands | grep -i sandbox`. Possible causes:
- Docker socket not mounted
- Permission denied on Docker socket
- Memory limit too low (increase `SANDBOX_DOCKER_ARGS`)

### Performance Issues

**Slow responses**

CPU inference is inherently slower than GPU. First-token latency is higher for CPU-optimized models. Smaller models are faster. Check resource usage with `docker stats`.

**Agent tasks fail or produce poor results**

Possible causes:
- Model lacks coding capability (try a model optimized for code)
- Context window too small (increase `LLM_MAX_INPUT_TOKENS`)
- Output limit too low (increase `LLM_MAX_OUTPUT_TOKENS`)
- Temperature too low (try `0.1` or `0.2`)

### Network Issues

**Containers cannot communicate**

Check `docker network inspect ovms-net`. Verify both containers use `ovms-net` and that OpenHands expects the `ovms-llm` hostname.

### Getting Help

- [OpenHands documentation](https://docs.all-hands.dev/)
- [OVMS documentation](https://github.com/openvinotoolkit/model_server)


## References

- [OpenHands Project](https://github.com/All-Hands-AI/OpenHands)
- [OpenHands Documentation](https://docs.all-hands.dev/)
- [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
- [OVMS Documentation](https://github.com/openvinotoolkit/model_server/tree/main/docs)
- [Hugging Face Models](https://huggingface.co/models)
- [OpenAI API Specification](https://platform.openai.com/docs/api-reference)

### Related OVMS Demos

- [integration_with_OpenWebUI](../integration_with_OpenWebUI/) — General model interface integration
- [llm_standalone_flow](../llm_standalone_flow/) — Standalone LLM deployment

### Model Documentation

- [Qwen Models](https://huggingface.co/Qwen)
- [Llama Models](https://huggingface.co/meta-llama)
- [Mistral Models](https://huggingface.co/mistralai)
