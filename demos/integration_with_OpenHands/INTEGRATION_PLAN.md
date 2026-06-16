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
- reproduce the setup primarily through clear documentation.

## Core Philosophy: Documentation-First

This demo is **documentation-first**. The README.md is the authoritative source
of truth and must contain all instructions necessary for a user to understand and
reproduce the setup manually.

**Helper artifacts are provided for convenience only, not as dependencies:**

- **docker-compose.yml** — A reference configuration showing the service
  architecture. The README explains each section and maps it to manual Docker
  commands so users understand every step independently of Compose.

- **scripts/deploy_model_ovms.sh** — A convenience helper that automates
  repetitive tasks (model selection, tool parser configuration, health checks).
  The README documents exactly what the script does internally.

**Long-term direction:** These helper artifacts may be removed if OVMS
maintainers prefer a pure documentation-based demo. The README must remain
complete and useful without them, but the artifacts are provided as helpful
references for users who prefer them.

**Model storage:** Models, OpenVINO IR files, and downloaded artifacts are stored
externally to the Git repository. The recommended workspace is:

```text
${HOME}/ovms-openhands/
└── models/
    └── <model-id>/
        ├── openvino_model.xml
        ├── openvino_model.bin
        └── graph.pbtxt
```

The repository should never contain model files. When using the OVMS `--source_model`
workflow, OVMS handles model retrieval and graph generation automatically. The
README explains where files are created and how they are mounted.

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

## Updated Migration Direction

The upstream demo follows a **documentation-first approach**:

- **README.md** is the authoritative source of truth and primary deliverable (Section 1-4 implemented)
- **docker-compose.yml** is a reference configuration showing the service architecture
- **scripts/deploy_model_ovms.sh** is a convenience helper that automates repetitive tasks
- Models are stored externally to the Git repository
- Benchmarking, telemetry, and experimental tooling remain in the standalone prototype repository

**Key principle:** Users should be able to understand and reproduce the setup by
following the README alone. Helper artifacts are provided for convenience and
reference.

**Long-term vision:** The helper artifacts (Compose and script) may be removed
if OVMS maintainers prefer a pure documentation-based demo. The README must
remain complete and useful without them, but the artifacts are helpful references
for users who prefer them.

**Current status:** README Sections 1-4 (Overview, Architecture, Prerequisites, Preparing the Model) have been implemented. The documentation now provides a complete foundation for understanding the integration and preparing models. Remaining sections (Quick Start, Verification, Troubleshooting, References) will be implemented in the next phase.

## Agreed Target Directory Structure

This is the current architectural target for the upstream demo.

```text
demos/
└── integration_with_OpenHands/
  ├── README.md                    # Primary documentation (authoritative)
  ├── docker-compose.yml           # Reference configuration (convenience)
  ├── scripts/
  │   └── deploy_model_ovms.sh    # Convenience helper (not required)
  └── screenshots/                 # Visual verification guide
```

**Model storage (external to Git repository):**
```text
${HOME}/ovms-openhands/
└── models/
    └── <model-id>/               # Downloaded/pulled OpenVINO models
        ├── openvino_model.xml
        ├── openvino_model.bin
        └── graph.pbtxt
```

**What is NOT included in upstream demo:**
- Benchmarking code
- Telemetry collection
- Experimental configurations
- Model caches or artifacts in the repository
- Prototype-specific development tooling
- `.env.example` or `.gitignore` (environment variables are used directly; models are external)

## Component Roles

### README.md (Authoritative)

The README is the primary deliverable and must be complete. It explains:

- Architecture overview with diagrams
- How OVMS and OpenHands interact over the OpenAI API
- Prerequisites (Docker, HF_TOKEN, hardware)
- Model preparation options (OVMS `--pull` workflow)
- Manual deployment steps (Docker commands without Compose)
- docker-compose.yml reference (what each section does)
- deploy_model_ovms.sh reference (what the script automates)
- Verification workflow (curl tests, OpenHands agent task)
- Troubleshooting common issues
- Supported model families and tool parser requirements

**Key principle:** A user should be able to set up the integration by reading
the README alone, without using the helper artifacts.

### docker-compose.yml (Static Reference Configuration)

Provided as a static reference implementation showing the complete service
architecture. The file:

- Uses Docker Compose environment variable substitution for runtime configuration
- Consumes environment variables exported by the helper script or set manually
- Relies on OVMS `--source_model` workflow for model download and graph generation
- Does NOT require runtime patching or placeholder replacement
- Serves as a reference implementation of the manual Docker commands

The README:
- Explains each service (ovms-llm, openhands)
- Maps Compose sections to equivalent Docker CLI commands
- Documents the required environment variables and their purposes
- Explains volume mounts and networking
- Shows how to achieve the same result without Compose

**Status:** Reference and convenience artifact. May be removed in favor of pure
documentation without reducing the value of the README.

### scripts/deploy_model_ovms.sh (Convenience Helper)

Automates repetitive tasks for user convenience:

- Parses model and deployment arguments (model_id, device, parser)
- Validates prerequisites (Docker, docker compose, HF_TOKEN)
- Normalizes model names for filesystem safety
- Resolves tool parsers based on model family
- Prepares external model workspace (`${HOME}/ovms-openhands/models`)
- Exports runtime environment variables for docker-compose.yml
- Launches `docker compose up -d` with the static compose file
- Waits for OVMS health via `/v1/config` polling
- Prints diagnostics and equivalent manual workflow

The script:
- Does NOT patch or modify docker-compose.yml at runtime
- Does NOT rewrite configuration files
- Is a thin wrapper around the manual README workflow
- Serves as a reference for users who prefer automation

The README documents exactly what this script does so users understand the
automation or can perform steps manually.

**Status:** Convenience and reference artifact. May be removed in favor of pure
documentation without reducing the value of the README.

### Prototype Repository Retains

The standalone prototype repository continues to host:

- Benchmark harness and comparison tools
- Telemetry collection and log parsing
- Historical experiments and configurations
- Development/validation workflows

## Recommended User Workflow

The README should present two equivalent paths to the same result:

### Path A: Using Helper Artifacts (Convenience)

1. **Configure prerequisites** — Set `HF_TOKEN` in environment
2. **Run deployment script** — `./scripts/deploy_model_ovms.sh <model_id>`
3. **Wait for successful deployment** — Script confirms OVMS health
4. **Verify OVMS** — Direct `curl` test to `/v3/chat/completions`
5. **Use OpenHands** — Open web UI, create agent task

### Path B: Manual Setup (Documentation-Driven)

1. **Configure prerequisites** — Set `HF_TOKEN` in environment
2. **Prepare model workspace** — Create external model directory
3. **Deploy OVMS** — Use Docker CLI or modified Compose
4. **Verify OVMS** — Direct `curl` test
5. **Configure OpenHands** — Set LLM_BASE_URL and LLM_MODEL
6. **Use OpenHands** — Open web UI, create agent task

**Both paths achieve the same result.** The helper artifacts automate repetitive
steps but are not required. The README must document both approaches.

## Implementation Roadmap

### Phase 1: Project Skeleton
- [x] Create upstream demo directory structure.
- [x] Create README skeleton.
- [ ] Add placeholder for screenshots (to be captured during validation).

### Phase 2: Helper Artifacts (Reference)
- [x] Implement docker-compose.yml as static reference configuration.
- [x] Implement scripts/deploy_model_ovms.sh as convenience helper.

### Phase 3: Documentation (Primary Deliverable)
- [x] Expand README with architecture overview and diagrams.
- [ ] Document manual setup workflow (Docker CLI commands).
- [ ] Document docker-compose.yml reference (what each section does).
- [x] Document deploy_model_ovms.sh reference (what the script does) - conceptual overview in Section 4
- [x] Ensure README can standalone without helpers - reinforced in Overview and Section 4
- [ ] Add verification workflow (curl tests, OpenHands agent task).
- [ ] Add troubleshooting section.
- [x] Document model families and tool parser requirements - included in Section 4
- [ ] Capture screenshots for visual verification.

### Phase 4: Validation
- [ ] Test manual setup workflow (README-only, no helpers).
- [ ] Test helper artifact workflow (Compose + script).
- [ ] Fresh clone validation.
- [ ] OVMS API verification.
- [ ] OpenHands agent task verification.
- [ ] Final documentation review for completeness.

## Migration Log

This section tracks significant decisions and changes to the integration plan.

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

### Step 1: Architecture Re-evaluation (Initial Approach)

- Reviewed `deploy_model_ovms.sh` and identified its encapsulation of non-trivial
  operational knowledge (tool parser resolution, model normalization, health-wait
  logic).
- Agreed on hybrid architecture: `docker-compose.yml` as scaffolding +
  `deploy_model_ovms.sh` as deployment helper.
- Documented component responsibilities and recommended user workflow.

### Step 2: Philosophy Re-Alignment (OVMS Mentor Feedback)

**Major directional change based on OVMS maintainer feedback:**

- Shifted to **documentation-first philosophy** — README.md is the authoritative
  source of truth, not the helper scripts.
- **Downgraded helper artifacts:**
  - `docker-compose.yml` → Reference configuration only
  - `deploy_model_ovms.sh` → Convenience helper only
- **Established external model storage:** Models stored in user-local directory
  (e.g., `${HOME}/ovms-openhands/models`), never in the Git repository.
- **Clarified long-term direction:** Helper artifacts may be removed; README must
  be complete without them.
- **Updated repository structure:** Simplified to README, reference configs,
  convenience script, and screenshots only.
- **No migration of:** Benchmarking code, telemetry, experiments, or model caches.

**Implementation impact:**
- README must document manual setup workflow equivalent to using helpers.
- docker-compose.yml is reference material, not a required component.
- deploy_model_ovms.sh is optional automation, not the primary interface.
- Users should be able to succeed by following README documentation alone.

### Step 3: Architectural Simplification

Simplified the deployment workflow by removing runtime docker-compose patching:

**Design changes:**
- `docker-compose.yml` is now a static reference configuration.
- Runtime configuration uses Docker Compose environment variable substitution.
- The helper script exports environment variables rather than patching YAML.
- OVMS `--source_model` handles model download and graph generation.

**Rationale:**
- Better aligns with the documentation-first philosophy.
- Keeps helper artifacts as optional convenience implementations.
- Makes the manual workflow more transparent (no hidden patching logic).
- Leverages Docker Compose native capabilities instead of custom logic.

**Implementation:**
- Replaced compose placeholders with `${VAR}` environment variable references.
- Updated `deploy_model_ovms.sh` to export runtime configuration.
- Removed any in-place file modification or patching logic.
- Script now prints the manual equivalent for user transparency.

### Phase 1: Project Skeleton (Completed)

Architectural milestone:
- Created upstream demo directory structure at `model_server/demos/integration_with_OpenHands/`.
- Established documentation-first philosophy and component roles.
- Added `README.md` with skeleton section headings.
- Prepared `scripts/` and `screenshots/` directories for later implementation.
- Set foundation for Phase 2 (helper artifacts) and Phase 3 (documentation expansion).

### Phase 2: Helper Artifacts (Completed)

Implementation milestone:
- Implemented `docker-compose.yml` as static reference configuration using environment
  variable substitution for runtime configuration.
- Implemented `scripts/deploy_model_ovms.sh` as optional convenience helper that exports
  runtime configuration and launches the static compose file.
- Adopted OVMS `--source_model` workflow for automatic model download and graph generation.
- Removed runtime docker-compose patching in favor of Docker Compose native substitution.
- Established external model storage at `${HOME}/ovms-openhands/models`.
- Set foundation for Phase 3 (README documentation).

### Phase 3: README Documentation - Sections 1-4 (Completed)

Documentation milestone:
- Implemented README Section 1 (Overview) with introduction to OpenHands, OVMS backend suitability,
  and documentation-first philosophy statement.
- Implemented README Section 2 (Architecture) with ASCII diagram, component descriptions,
  request flow explanation, and OpenHands-specific configuration requirements.
- Implemented README Section 3 (Prerequisites) with system requirements, network/port usage table,
  and external model storage location documentation.
- Implemented README Section 4 (Preparing the Model) with model selection guidance, tool parser
  explanation, `--source_model` workflow documentation, and conceptual overview of helper script behavior.

**Design principles preserved:**
- README.md remains the authoritative source of truth; helper artifacts are documented as optional.
- No new mandatory dependencies or unsupported workflows were introduced.
- Helper scripts are presented as convenience tools rather than required setup mechanisms.
- All commands, environment variables, and workflows are consistent with existing implementation.

**Consistency with implementation:**
- Environment variables (MODEL_ID, LOCAL_NAME, TARGET_DEVICE, TOOL_PARSER, MODEL_CACHE_DIR, HF_TOKEN)
  match docker-compose.yml and deploy_model_ovms.sh.
- Port documentation (8000, 9000, 3000) matches the compose configuration.
- Model storage location (${HOME}/ovms-openhands/models) matches the helper script defaults.
- Tool parser mappings (Qwen→qwen, Llama3/Mistral→hermes3) match the script resolution logic.
- OVMS `--source_model` workflow is documented as the recommended approach.

**Foundation for next phase:**
- README structure is now substantially expanded beyond skeleton.
- Architecture, prerequisites, and model preparation are fully documented.
- Remaining sections (Quick Start, Verification, Troubleshooting, References) are ready for implementation.
