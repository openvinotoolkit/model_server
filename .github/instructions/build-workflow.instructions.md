---
description: "Build workflow, Docker setup, Makefile targets, style checks and test execution for OVMS"
---
# OVMS Build & Test Workflow

## Which build do I need?

- **Most code changes** (`.cpp`/`.hpp`/`BUILD` under `src/`): build & test **inside the existing `-build` container** with Bazel. This is fast and is the default — do NOT rebuild the Docker image.
- **`Dockerfile.*` or `Makefile` changes** (build environment, dependencies, packaging): a **full image build** is required (`make docker_build`).
- **Windows**: use the batch scripts in the repo root (see the Windows Builds section) — no Docker container.

## Docker-Based Development

Building and testing is done **inside a Docker `-build` container** with the repository mounted. Developers do not run Bazel on the host directly.

### Checking for existing build images

Before building a new image (which is time-expensive), check if one exists:
```bash
docker images | grep -- -build
```

### Finding a container with the current workspace mounted

**Do NOT run a `docker ps`/`docker inspect`/mount-path scan as the first
step.** That scan is expensive and must only ever be a last resort.

**Always start here instead** — check the AI agent's repo memory note (e.g.
`/memories/repo/build-container.md`) for a previously confirmed container
name for this workspace:
1. If a remembered container name exists, verify it's still usable:
   ```bash
   docker ps -a --filter name=<remembered_name> --format '{{.ID}}\t{{.Status}}'
   ```
   - If `Up`, use it directly — no further scanning: `docker exec -w /ovms <name> ...`.
   - If stopped, restart it (`docker start <name>`) and reuse it. Still no scan needed.
2. Only if the remembered container no longer exists at all, fall back to
   scanning by mount path for a running container:
   ```bash
   docker ps -q | xargs -I{} docker inspect {} --format '{{.ID}} {{range .Mounts}}{{.Source}}{{end}}' | grep "$(pwd)"
   ```
   If found, use `docker exec -it <container_id> bash` to enter it directly.

   If no running container matches, check stopped containers too:
   ```bash
   docker ps -aq | xargs -I{} docker inspect {} --format '{{.ID}} {{range .Mounts}}{{.Source}}{{end}}' | grep "$(pwd)"
   ```
   If found stopped, use `docker start -i <container_id>`.
3. **Immediately update the repo memory note** with the newly discovered
   container name/ID so the scan is never repeated needlessly.

### Never re-scan once a container is cached

Once a container name is cached in repo memory, treat it as valid for the
rest of the session **and** future sessions. Do not re-run the mount-path
scan "just to double check" — only re-scan if the cached container is
confirmed gone (not just stopped) via the `docker ps -a --filter name=...`
check above.

### Starting a build container

If a `-build` image exists, start a container with the repo mounted:
```bash
docker run -it -v $(pwd):/ovms \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    <image_name> bash
```

If a container from a previous session is available (`docker ps -a`), reuse it:
```bash
docker start -i <container>
# or
docker exec -it <container> bash
```

## Bazel Commands (inside build container)

### Key targets

| Target | Description |
|--------|-------------|
| `//src:ovms` | Main OVMS server binary |
| `//src:ovms_test` | C++ unit tests (gtest) |
| `//src:ovms_shared` | C API shared library (`libovms_shared.so`) |

### Build the server
```bash
bazel build //src:ovms
```

### Build and run unit tests
```bash
bazel build //src:ovms_test
bazel test --test_summary=detailed --test_output=streamed //src:ovms_test
```

### Running specific tests (preferred during development)
```bash
bazel test --test_summary=detailed --test_output=streamed --test_filter="SuiteName.TestName" //src:ovms_test
```
Always run targeted tests first; run the full suite only after targeted tests pass.

### Red Hat builds
Pass `--//:distro=redhat` to all Bazel commands:
```bash
bazel build --//:distro=redhat //src:ovms
```

### Linux build config
```bash
--config=mp_on_py_on
```

### Windows build config
```bash
--config=win_mp_on_py_off
# or with Python:
--config=win_mp_on_py_on
```

## Makefile Targets (Docker-based workflow)

| Target | Description |
|--------|-------------|
| `make docker_build` | Full build: builder image → package → release images |
| `make ovms_builder_image` | Build the `-build` Docker image |
| `make targz_package` | Extract `.tar.gz` release package |
| `make ovms_release_images` | Build CPU and GPU release Docker images |
| `make run_unit_tests` | Run C++ unit tests in the `-build` container |
| `make test_functional` | Run Python functional te| `make style` | All code style checks: spell, clang-format, cpplint, cppclean (see Style Checking section) |

### Red Hat build via Make
```bash
make docker_build BASE_OS=redhat
```

Supported `BASE_OS` values: `ubuntu24`, `ubuntu22`, `redhat`

## Dockerfile Stages

Both `Dockerfile.ubuntu` and `Dockerfile.redhat` use multi-stage builds:

| Stage | Purpose |
|-------|---------|
| `base_build` | System dependencies, Boost, Azure SDK, OpenCV |
| `build` | Bazel + OpenVINO setup, compiles OVMS (the `-build` container) |
| `capi-build` | Builds C API shared library and examples |
| `pkg` | Packages everything into a `.tar.gz` |
| `release` | Minimal runtime image with entrypoint |

## Style Checking

**Run style checks on the HOST, not inside the `-build` container.** The Makefile targets set up
their own pinned `venv-style` virtualenv with the correct tool versions; the `-build` container
does not have `clang-format`/`cpplint` installed. Never invoke `clang-format` or `cpplint` by hand
— always go through the `make` targets, or you get the wrong version and spurious diffs.

**Always run checks individually and sequentially** — never use `make style` (which runs all checks together and wastes time re-running already-passed steps). Fix each step before moving to the next:

1. **Spelling**:
   ```bash
   make spell
   ```
2. **clang-format** (formatting):
   ```bash
   make clang-format-check
   ```
3. **cpplint** (lint rules):
   ```bash
   make cpplint
   ```
4. **cppclean** (unused includes/code):
   ```bash
   make cppclean
   ```

Fix issues from step N before running step N+1 — later steps produce noise if formatting is off.

## Test Setup

Before running tests, prepare test models:
```bash
make prepare_models
```

If LLM test models need regeneration:
```bash
rm -rf src/test/llm_testing
make prepare_models
```

## Windows Builds

Windows builds use batch files in the repository root:
- `windows_install_build_dependencies.bat` — Install MSVC 2022 Build Tools, etc.
- `windows_build.bat` — Main build script
- `windows_test.bat` — Run tests

## Test Structure

- Unit tests are in `src/test/` — gtest-based C++ tests
- Test files: `*_test.cpp` naming convention
- Test utilities: `test_utils.hpp`, `light_test_utils.hpp`, `c_api_test_utils.hpp`
- Test models: `src/test/` subdirectories (`dummy/`, `passthrough/`, `summator/`)
- Specialized: `src/test/llm/`, `src/test/mediapipe/`, `src/test/python/`, `src/test/embeddings/`
