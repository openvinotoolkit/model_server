---
description: "Build workflow, Docker setup, Makefile targets, style checks and test execution for OVMS"
---
# OVMS Build & Test Workflow

## Docker-Based Development

Building and testing is done **inside a Docker `-build` container** with the repository mounted. Developers do not run Bazel on the host directly.

### Checking for existing build images

Before building a new image (which is time-expensive), check if one exists:
```bash
docker images | grep -- -build
```

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
| `make test_functional` | Run Python functional tests |
| `make style` / `make cpplint` / `make spell` | Code style checks (see Style Checking section) |

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

Run style checks **sequentially** — fix each step before moving to the next:

1. **clang-format** (formatting):
   ```bash
   make style
   ```
2. **cpplint** (lint rules):
   ```bash
   make cpplint
   ```
3. **Spelling**:
   ```bash
   make spell
   ```

Each step runs via Make. Fix issues from step N before running step N+1 — later steps produce noise if formatting is off.

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
