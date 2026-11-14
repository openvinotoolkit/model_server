# OpenHands Integration with OpenVINO Model Server {#ovms_demos_integration_with_openhands}

## Description

[OpenHands](https://github.com/All-Hands-AI/OpenHands) is an open-source software engineering agent that automates coding tasks through iterative LLM inference, tool execution, and runtime sandbox environments. Unlike simple chat interfaces, OpenHands maintains long-running conversations, creates code execution sandboxes, and performs multi-step problem solving.

This demo integrates OpenHands with [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) using OVMS's OpenAI-compatible REST API. It demonstrates how to deploy OVMS as a backend for OpenHands, enabling agent workflows on local hardware with OpenVINO-optimized models.

Quick start: run `./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov` to deploy both OVMS and OpenHands containers.

---

## Prerequisites

- **Host architecture:** x86_64
- **Operating system:** Linux with Docker Engine and Docker Compose
- **Memory:** Minimum 8GB RAM (16GB+ recommended)
- **Hugging Face token:** Required for gated models (set `HF_TOKEN`)

---

## Running the Deployment

1. **Clone and navigate:**
   ```bash
   git clone https://github.com/openvinotoolkit/model_server.git
   cd model_server/demos/integration_with_OpenHands
   ```

2. **Set Hugging Face token** (required for gated models):
   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. **Deploy (CPU):**
   ```bash
   ./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov
   ```

4. **Deploy (GPU):**
   ```bash
   ./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov --device GPU
   ```
   Requires OpenVINO-compatible Intel GPU runtime on the host. Same model works on CPU or GPU—only the `--device` flag changes.

The script validates the environment, downloads the model, generates `docker-compose.yml`, and starts both containers.

---

## Verifying the Deployment

### Stage 1: Verify OVMS

The deployment script sets the published OVMS REST port (default: 8000). Use this port for verification.

**Check health:**
```bash
curl -s http://localhost:${OVMS_REST_PORT:-8000}/v1/config
```

The response should include `"model_status": "AVAILABLE"`.

**Test a completion request:**
```bash
curl -X POST http://localhost:${OVMS_REST_PORT:-8000}/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your-model-name>",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }'
```

Replace `<your-model-name>` with the actual model name (e.g., `qwen3-8b-int8-ov` for the example above).

### Stage 2: Verify OpenHands

The deployment script sets the published OpenHands port (default: 3000).

1. **Open the web UI** at `http://localhost:${OPENHANDS_PORT:-3000}`

2. **Configure the OVMS-backed model:**
   - Click **Settings** → **LLM**
   - Enable **Advanced** mode if needed
   - Set **Custom Model:** `openai/<your-model-name>` (e.g., `openai/qwen3-8b-int8-ov`)
   - Set **Base URL:** `http://ovms-llm:8000/v3`
   - Set **API Key:** `unused`
   - Click **Save**

3. **Create an agent task:**
   ```
   Create a Python function that calculates the factorial of a number.
   ```

4. **Verify behavior:**
   - OpenHands creates a runtime sandbox container
   - The agent writes and tests code
   - OVMS logs show incoming `/v3/chat/completions` requests

---

## Container Management

Use `docker compose` for routine operations on the generated `docker-compose.yml`:
```bash
docker compose ps          # status
docker compose logs -f     # follow logs
docker compose restart     # restart all
docker compose restart openhands  # restart one service
docker compose down        # stop and remove
```

Run the deployment script again when changing model, device, or ports—it regenerates the compose file each time.

---

## References

- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
- [Hugging Face Models](https://huggingface.co/models)
- [OpenVINO Toolkit Models](https://huggingface.co/OpenVINO)