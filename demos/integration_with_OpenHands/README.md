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
| 8000 | OVMS      | OpenAI-compatible REST API  |
| 9000 | OVMS      | gRPC API (not used here)    |
| 3000 | OpenHands | Web UI                      |

Ensure these ports are available on your host.

---

## Preparing the Model

### Choosing a Compatible Model

OpenHands requires models with instruction-following capability, coding proficiency, sufficient context window (4096+ tokens), and tool calling support.

| Model Family | Tool Parser | Notes                                  |
|--------------|-------------|----------------------------------------|
| Qwen 3 Coder | `qwen3coder`| Strong coding performance, various sizes |
| Qwen 3       | `hermes3`   | Recommended for low end systems        |
| Llama 3      | `llama3`    | Good general instruction following   |
| Mistral      | `mistral`   | Efficient inference                  |

> **Note:** Examples use `OpenVINO/qwen3-0.6b-int8-ov`, a lightweight model suitable for validation. Other compatible models may also be used.

### Tool Parser Selection

Tool parsers enable structured output for function calling. When OpenHands executes a tool (running code, reading files), it expects the LLM to return structured JSON specifying the tool name and arguments. The tool parser converts model outputs into this format.

Without the correct tool parser, the model does not generate tool calls in the expected format, causing tool call extraction to fail.

For details on OVMS model retrieval and workspace layout, see [ADVANCED_DEPLOYMENT.md](ADVANCED_DEPLOYMENT.md).

---

## Deployment

This demo uses `docker-compose.yml` and `scripts/deploy_model_ovms.sh` to automate deployment. Clone the repository and navigate to the demo directory before proceeding.

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
   ./scripts/deploy_model_ovms.sh OpenVINO/qwen3-0.6b-int8-ov
   ```

   The script validates your environment, prepares the model, and launches both containers. See [ADVANCED_DEPLOYMENT.md](ADVANCED_DEPLOYMENT.md) for details on what the script does.

4. **Verify the deployment** (see next section)

**Optional parameters:**
```bash
# Specify device, parser, or cache directory
./scripts/deploy_model_ovms.sh OpenVINO/qwen3-0.6b-int8-ov \
    --device CPU \
    --parser hermes3 \
    --cache-dir ~/custom-models

# Skip health check for faster feedback
./scripts/deploy_model_ovms.sh OpenVINO/qwen3-0.6b-int8-ov --skip-wait
```

For manual Docker deployment, see [ADVANCED_DEPLOYMENT.md](ADVANCED_DEPLOYMENT.md).

---

## Verifying the Deployment

Verify the integration in two stages: first OVMS directly, then OpenHands.

### Stage 1: Verify OVMS

**Check health:**
```bash
curl -s http://localhost:8000/v1/config | jq .
```

The response should include `"model_status": "AVAILABLE"`.

**Test a completion request:**
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-int8-ov",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }'
```

If OVMS fails to respond, check `docker logs ovms-llm`, verify the model downloaded at `$MODEL_CACHE_DIR`, and ensure `HF_TOKEN` is set if needed.

### Stage 2: Verify OpenHands

1. **Open the web UI** at `http://localhost:3000`
   ![OpenHands UI](<screenshots/Pasted image (2).png>)

2. **Configure the OVMS-backed model:**
   - Click **Settings** → **LLM**
   - Enable **Advanced** mode if needed
   - Set **Custom Model:** `openai/qwen3-0.6b-int8-ov`
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

**Model status is not `AVAILABLE`**

Check `curl -s http://localhost:8000/v1/config`. Possible causes:
- Model still downloading (wait longer for large models)
- Out of memory (check host RAM; model may be too large)
- Tool parser mismatch (verify `TOOL_PARSER` matches model family)

**Connection refused**

Possible causes:
- OVMS container not running (`docker ps`)
- Wrong port (verify `8000:8000` mapping)
- Firewall blocking port 8000

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
