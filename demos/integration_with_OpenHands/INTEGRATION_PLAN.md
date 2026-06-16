# OpenHands Integration Plan

## High-Level Goal

The goal of this integration is to provide an upstream-quality OVMS demo showing
how to run OpenHands with an OpenVINO Model Server backend. The final demo should
let a user:

- deploy a suitable text-generation model with OVMS;
- connect OpenHands to the OVMS OpenAI-compatible REST endpoint;
- verify the connection with a direct API request and an OpenHands agent task;
- understand the model, context-window, tool-calling, networking, and resource
  requirements that affect agent behavior; and
- reproduce the setup primarily through documented Docker configuration and
  environment variables.

The deliverable should follow the lightweight, task-oriented style of
`integration_with_OpenWebUI`. It should not transplant the prototype repository
or become a separate application or benchmark framework inside OVMS.

## Architecture of `integration_with_OpenWebUI`

The OpenWebUI demo is primarily a documentation integration rather than a
custom software component. Its directory contains one comprehensive `README.md`
and screenshots demonstrating configuration and results.

Its core architecture is:

```text
User
  |
  v
Open WebUI
  |
  | OpenAI-compatible HTTP requests
  | Base URL: http://localhost:8000/v3
  v
OpenVINO Model Server
  |
  v
OpenVINO-format generative models
```

Key characteristics:

- OVMS and the client remain independent components connected through standard
  REST APIs.
- Models are prepared with OVMS-native `--pull` and `--add_to_config` commands.
- Linux instructions use the published OVMS Docker image; Windows instructions
  use the OVMS binary.
- Each capability begins with model deployment and a direct `curl` verification,
  followed by client configuration and a demonstrated user workflow.
- Client-specific behavior is configured through its UI or supported settings,
  without patching OpenWebUI or OVMS source code.
- The README is organized as a guided recipe with prerequisites, numbered setup
  steps, references, notes, and screenshots.
- The demo expands from basic chat into optional OVMS-backed capabilities such
  as RAG, image generation, VLM, tools, web search, memory, code execution, and
  audio.

This establishes the preferred philosophy for the OpenHands demo: use existing
published components, keep the API boundary explicit, make configuration
reproducible, verify each layer independently, and document the workflow
visually.

## Architecture of the Prototype Repository

The `openhands-openvino-integration` repository is a local development,
validation, and benchmarking environment. It contains more machinery than an
OVMS demo should ultimately require.

Its functional runtime architecture is:

```text
User or benchmark runner
  |
  v
OpenHands web application and conversation API
  |
  | OpenAI-compatible chat completions
  | LLM_BASE_URL=http://ovms-llm:8000/v3
  | LLM_MODEL=openai/<served-model-name>
  v
OVMS container
  |
  | Text-generation pipeline and model-specific tool parser
  v
OpenVINO model

OpenHands also creates separate runtime sandbox containers for agent actions.
```

The main components are:

- `docker-compose.yml`, which defines OVMS and OpenHands services on a shared
  Docker network, publishes ports `8000`, `9000`, and `3000`, passes OpenHands
  LLM settings through environment variables, mounts the Docker socket for
  runtime sandboxes, and persists OpenHands settings.
- OVMS configuration and model graph files under `configs/`, used by some of
  the prototype's deployment paths.
- deployment helpers that either download/convert models and generate a
  MediaPipe graph or use the newer OVMS-native `--source_model` workflow;
- an OpenHands startup helper that handles persisted settings, networking, and
  cleanup of old runtime containers;
- direct endpoint validation scripts; and
- a Python benchmark harness that drives the OpenHands conversation API,
  measures readiness and completion, scores responses, collects Docker
  telemetry and logs, and writes reproducibility artifacts.

The prototype demonstrates that OpenHands can route requests to OVMS through
the `/v3/chat/completions` endpoint. It also records practical findings:

- OpenHands requires the provider prefix in model configuration, for example
  `openai/<model-name>`, while OVMS receives the served model name.
- A non-empty placeholder API key is needed by the client even though OVMS does
  not authenticate the request.
- A shared Docker network and the OVMS service name are more stable than
  container IP addresses.
- Persisted OpenHands settings can override environment variables.
- Agent prompts need a large context window and a capable coding/instruction
  model; small models may connect successfully but fail at useful agent tasks.
- Each OpenHands conversation may create a resource-consuming runtime sandbox.
- Tool-parser selection, generation limits, timeouts, and host memory are
  important parts of a usable deployment.

The repository also contains historical experiments and configurations for
different models and deployment methods. Some checked-in names, paths, device
choices, and compatibility conclusions do not describe one single canonical
configuration. They should therefore be treated as evidence and lessons, not
copied verbatim into the OVMS demo.

## Initial Similarities and Differences

### Similarities

- Both integrations connect an independent user application to OVMS over an
  OpenAI-compatible HTTP API.
- Both use the OVMS `/v3` base path and require an explicitly configured model
  name.
- Both can run with published Docker images and avoid changes to either
  application's source code.
- Both benefit from validating OVMS directly before debugging the client.
- Both need model-specific deployment parameters and enough host resources for
  the selected model.
- Both are best presented as reproducible setup and configuration recipes.

### Differences

- OpenWebUI is a general model interface, while OpenHands is an autonomous
  coding-agent application that adds long system prompts, iterative inference,
  tool/action behavior, and runtime sandbox containers.
- The OpenWebUI demo is documentation-first and has no orchestration or
  benchmark code. The prototype is an engineering workbench with Compose,
  deployment scripts, tests, telemetry, benchmarks, and extensive investigation
  notes.
- OpenWebUI supports many independent generative use cases. The initial
  OpenHands demo should focus narrowly on reliable text-generation and coding
  agent behavior.
- OpenHands has client-specific configuration requirements, including the
  `openai/` model prefix, a placeholder API key, Docker socket access, persistent
  settings, and stable container networking.
- Successful API connectivity is sufficient for a basic OpenWebUI chat demo,
  but it is not sufficient for OpenHands: model quality, context capacity,
  structured tool output, latency, and sandbox resources determine whether an
  agent task actually succeeds.
- The prototype includes legacy and experimental model-serving paths. The OVMS
  demo should prefer current OVMS-native model preparation and a small number of
  clearly documented configuration choices.

## Initial Migration Direction

The upstream demo follows a hybrid approach:

- **docker-compose.yml** provides clean service scaffolding and serves as a patch target
- **deploy_model_ovms.sh** is a first-class deployment helper that encapsulates tool-parser resolution, model normalization, compose patching, and health-wait logic
- **README.md** is the primary user interface, guiding users through the recommended workflow
- Benchmarking, telemetry, and experimental tooling remain in the standalone prototype repository

This balances upstream simplicity (clean, readable Compose) with operational knowledge preservation (script encodes hard-won lessons).

## Agreed Target Directory Structure

This is the current architectural target for the upstream demo.

```text
demos/
└── integration_with_OpenHands/
  ├── README.md                    # Primary user documentation
  ├── docker-compose.yml           # Infrastructure scaffolding & patch target
  ├── .env.example                 # Configuration template (HF_TOKEN, etc.)
  ├── scripts/
  │   ├── deploy_model_ovms.sh    # First-class deployment helper
  │   └── start_openhands.sh      # Optional lightweight helper (TBD in implementation)
  ├── screenshots/                 # Visual verification guide
  └── .gitignore
```

## Component Responsibilities

**docker-compose.yml:**
- Service scaffolding (ovms-llm, openhands)
- Image definitions, port publishing, network definition
- Docker socket mount for OpenHands runtime sandboxes
- Persistent volume for OpenHands settings
- Serves as a patch target for `deploy_model_ovms.sh`

**deploy_model_ovms.sh:**
- Model family normalization (HF model ID → local name)
- Tool parser resolution (model-family → parser mapping)
- `ovms_config.json` generation with aligned paths
- `docker-compose.yml` patching (command block, LLM_MODEL)
- HF_TOKEN injection into OVMS container environment
- Deployment orchestration (`docker compose up -d`)
- Health-wait logic with dual signals (logs + REST endpoint)
- Diagnostics on failure

**README.md:**
- Architecture overview and diagram
- Prerequisites (Docker, HF_TOKEN)
- Recommended workflow (step-by-step)
- Verification and troubleshooting
- Supported model families reference

**start_openhands.sh (optional, TBD during implementation):**
- Lightweight OpenHands launcher
- Must not duplicate deployment logic from `deploy_model_ovms.sh`
- Exact responsibilities to be determined during implementation review

**Prototype repository retains:**
- Benchmark harness and comparison tools
- Telemetry collection and log parsing
- Historical experiments and alternative configurations

## Recommended User Workflow

1. **Configure prerequisites** — Set `HF_TOKEN` in `.env` or environment
2. **Run deployment script** — `./scripts/deploy_model_ovms.sh <model_id>`
3. **Wait for successful deployment** — Script confirms OVMS health
4. **Verify OVMS** — Direct `curl` test to `/v3/chat/completions`
5. **Use OpenHands** — Open web UI, create agent task

This workflow preserves the OpenWebUI demo's documentation-first philosophy while using the deployment script to encapsulate non-trivial operational knowledge.

## Implementation Roadmap

### Phase 1: Project Skeleton
- [x] Create upstream demo directory structure.
- [x] Create README skeleton.
- [x] Add .env.example.
- [x] Add .gitignore.
- [x] Create screenshots directory.

### Phase 2: Deployment
- [ ] Port and simplify docker-compose.yml (service scaffolding, patch target).
- [ ] Port and simplify deploy_model_ovms.sh (tool parser logic, patching, health-wait).
- [ ] Review start_openhands.sh responsibilities and simplify if included (optional, must not duplicate deploy_model_ovms.sh).

### Phase 3: Documentation
- [ ] Expand README with architecture overview.
- [ ] Add setup instructions.
- [ ] Add verification workflow.
- [ ] Add troubleshooting section.
- [ ] Capture screenshots.

### Phase 4: Validation
- [ ] Fresh clone validation.
- [ ] docker compose up validation.
- [ ] OVMS API verification.
- [ ] OpenHands agent task verification.
- [ ] Final documentation review.

## Migration Log

This section will be updated after every future migration step.

### Step 0: Architecture Inspection and Planning

- Inspected `model_server/demos/integration_with_OpenWebUI`.
- Inspected the structure, runtime configuration, documentation, deployment
  helpers, validation utilities, and benchmark design in
  `openhands-openvino-integration`.
- Identified the OpenWebUI demo as the canonical documentation-first pattern.
- Identified the prototype's validated runtime boundary and operational lessons,
  while noting that its historical configurations should not be migrated as a
  single package.
- Created this planning document. No implementation code or configuration was
  added.

### Step 1: Architecture Re-evaluation After deploy_model_ovms.sh Review

- Reviewed `deploy_model_ovms.sh` and identified its encapsulation of non-trivial operational knowledge (tool parser resolution, model normalization, compose patching, health-wait logic).
- Re-evaluated initial assumption of static `docker-compose.yml` approach.
- Agreed on hybrid architecture: `docker-compose.yml` as clean scaffolding + `deploy_model_ovms.sh` as first-class deployment helper.
- Documented component responsibilities and recommended user workflow.
- Updated `INTEGRATION_PLAN.md` to reflect agreed design decisions. No implementation code added.

### Phase 1: Project Skeleton (Completed)

- Created upstream demo directory structure at `model_server/demos/integration_with_OpenHands/`.
- Added `README.md` with skeleton section headings (no content yet).
- Added `.env.example` with HF_TOKEN and optional MODEL_ID template.
- Added `.gitignore` for `.env`, screenshots, and `*.log` files.
- Initialized `scripts/` and `screenshots/` directories (with `.gitkeep`).
- Marked Phase 1 checklist items as complete.
