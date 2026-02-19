# Copilot Instructions for OpenVINO Model Server (OVMS)

## Project Overview

OpenVINO Model Server (OVMS) is a high-performance inference serving platform built on top of **OpenVINO** and **OpenVINO GenAI**. The codebase is primarily **C++** with **Bazel** as the build system. Supporting infrastructure uses **Makefiles**, **Dockerfiles** (Ubuntu & Red Hat), and **batch files** (Windows).

**Performance is a top priority** — both **throughput** and **latency** are critical. Code changes should be evaluated for their performance impact. Avoid unnecessary copies, allocations, and blocking operations on the hot path.

## Repository Structure

- `src/` — Main C++ source code (server, gRPC/REST handlers, model management, pipelines, mediapipe, LLM, C API)
- `src/test/` — C++ unit tests (gtest-based); this is where most developer tests live
- `src/python/` — Python bindings and related code
- `demos/` — End-user demo applications
- `client/` — Client libraries (C++, Python, Go, Java)
- `docs/` — Documentation
- `third_party/` — Third-party dependency definitions for Bazel
- `Dockerfile.ubuntu` / `Dockerfile.redhat` — Multi-stage Dockerfiles for Linux builds
- `Makefile` — Orchestrates Docker-based builds and test runs
- `*.bat` files — Windows build and setup scripts

## Code Style

- C++ style is enforced via `cpplint` and `clang-format`
- Run `make style` to check formatting
- Apache 2.0 license headers are required on all source files

## Expertise Areas

1. **OpenVINO Expertise:**
   - Proficient with OpenVINO core libraries and `ov::genai` components
   - Familiar with OpenVINO performance optimization techniques
2. **C++ Proficiency:**
   - Strong C++17 skills
   - Familiar with best practices in memory management, concurrency, and template programming
3. **Serving Infrastructure:**
   - gRPC and REST API handler design
   - Model management, pipeline orchestration, and MediaPipe integration
   - C API (`libovms_shared.so`) surface and bindings
4. **Build System Awareness:**
   - Bazel build configuration and dependency management
   - Minimizing C++ build times: forward declarations, include-what-you-use, avoiding transitive header leakage
   - Understanding of Bazel targets, build flags (`--//:distro`), and multi-stage Docker builds

## Code Review Instructions for PRs

When analyzing a Pull Request, follow this protocol:

1. Follow **C++ Core Guidelines** strictly. Include references in review comments.
2. Check for **hidden performance costs**: avoid `dynamic_cast` on the hot path; suggest `static_cast` or redesign if the type is known.
3. **Avoid copies**: ensure large data structures (tensors, buffers) are passed by reference or moved, not copied.
4. **Documentation**: ensure new public APIs have docstrings in C++ headers and Python bindings; update `docs/` as needed.
5. **Test coverage**: ensure that new features or changes have corresponding tests in `src/test/`.
6. **Formatting & safety:**
   - No `using namespace std; using namespace ov;`. Prefer explicit using with specific symbols if needed, for readability.
   - No `auto` for primitive types where it obscures readability.
   - Use `const` and `constexpr` wherever possible.
7. Pass non-fundamental values by `const` reference wherever possible.
8. Prefer member initializer lists over direct assignments in constructor bodies.
9. Verify that the result of every newly introduced function is used in at least one call site (except `void` functions).
10. Use descriptive function and variable names. Avoid duplicate code — extract common functionality into reusable utilities.
11. When initial container values are known upfront, prefer initializer-list / brace-initialization over constructing an empty container and inserting.
12. Unused functions and includes are not allowed. Build times are already long — do not add unnecessary `#include` directives. Prefer forward declarations where possible and follow the include-what-you-use principle.
    - **Forward-declare in headers, include in `.cpp`**: if a header only uses pointers or references to a type, use a forward declaration (`class Foo;`) instead of `#include "foo.hpp"`. Move the full `#include` to the `.cpp` file where the type is actually used.
    - **Keep headers self-contained but minimal**: each header must compile on its own, but should not pull in transitive dependencies that callers don't need.
    - **Prefer opaque types / Pimpl**: for complex implementation details, consider the Pimpl idiom to keep implementation-only types out of the public header entirely.
    - **Never include a header solely for a typedef or enum**: forward-declare the enum (`enum class Foo;` in C++17) or relocate the typedef to a lightweight `fwd.hpp`-style header.
13. Be mindful when accepting `const T&` in constructors or functions that store the reference: verify that the referenced object's lifetime outlives the usage to avoid dangling references.





## Build System

### Bazel (primary build tool)

Building and testing is done **inside a Docker `-build` container** with the repository mounted. Developers do not run Bazel on the host directly.

**Important:** Building the `-build` image from scratch is time-expensive, but may be required if dependencies or the build environment change. Before building a new one, check if one already exists:
```bash
docker images | grep -- -build
```
If a `-build` image exists, start a container from it with the repository mounted:
```bash
docker run -it -v $(pwd):/ovms \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    <image_name> bash
```
If a container from a previous session is still available (`docker ps -a`), reuse it with `docker start -i <container>` or `docker exec -it <container> bash`.

**Key Bazel targets:**

| Target | Description |
|--------|-------------|
| `//src:ovms` | Main OVMS server binary |
| `//src:ovms_test` | C++ unit tests (gtest) |
| `//src:ovms_shared` | C API shared library (`libovms_shared.so`) |
| `//src/python/binding:test_python_binding` | Python binding tests |

**Build the server:**
```bash
bazel build //src:ovms
```

**Build and run unit tests:**
```bash
bazel build //src:ovms_test
bazel test --test_summary=detailed --test_output=streamed //src:ovms_test
```

### Red Hat builds — distro flag

For Red Hat (UBI9) builds, the `--//:distro=redhat` flag must be passed to all Bazel commands:
```bash
bazel build --//:distro=redhat //src:ovms
bazel test --//:distro=redhat //src:ovms_test
```

The default distro is `ubuntu`. Inside the `-build` container, the distro is auto-detected from `/etc/redhat-release`.

### Makefile (Docker-based workflow)

The Makefile orchestrates full Docker-based builds. Key targets:

| Target | Description |
|--------|-------------|
| `make docker_build` | Full build: builder image → package → release images (default target) |
| `make ovms_builder_image` | Build the `-build` Docker image (compilation container) |
| `make targz_package` | Extract `.tar.gz` release package |
| `make ovms_release_images` | Build CPU and GPU release Docker images |
| `make run_unit_tests` | Run C++ unit tests in the `-build` container |
| `make test_functional` | Run Python functional tests |
| `make style` / `make cpplint` | Code style checks |

**Red Hat build via Make:**
```bash
make docker_build BASE_OS=redhat
```

**Supported `BASE_OS` values:** `ubuntu24`, `ubuntu22`, `redhat`

### Dockerfile stages

Both `Dockerfile.ubuntu` and `Dockerfile.redhat` use multi-stage builds:

| Stage | Purpose |
|-------|---------|
| `base_build` | System dependencies, Boost, Azure SDK, OpenCV |
| `build` | Bazel + OpenVINO setup, compiles OVMS (the `-build` container) |
| `capi-build` | Builds C API shared library and examples |
| `pkg` | Packages everything into a `.tar.gz` |
| `release` | Minimal runtime image with entrypoint |

### Windows builds

Windows builds use batch files in the repository root:
- `windows_install_build_dependencies.bat` — Install build dependencies (MSVC 2022 Build Tools, etc.)
- `windows_build.bat` — Main build script
- `windows_test.bat` — Run tests

Windows-specific Bazel config: `--config=win_mp_on_py_off` (or `--config=win_mp_on_py_on` for Python support).

## Testing

### Test setup

Before running tests, test models must be prepared:
```bash
make prepare_models
```

Models are exported using the `demos/common/export_models/export_model.py` script (used internally by the test setup).

If LLM test models change (e.g., new model version, OpenVINO version change or config update), you may need to remove and regenerate the LLM test data:
```bash
rm -rf src/test/llm_testing
make prepare_models
```

### Running tests

Running the full `//src:ovms_test` suite is **time-consuming**. During development, always run only the test fixtures relevant to your changes first using `--test_filter`:
```bash
bazel test --test_summary=detailed --test_output=streamed --test_filter="SuiteName.TestName" //src:ovms_test
```
Run the full test suite only after the targeted tests pass.

### Test structure

- **Unit tests** are in `src/test/` — gtest-based C++ tests covering all server components
- Test files follow the `*_test.cpp` naming convention
- Test utilities: `test_utils.hpp`, `light_test_utils.hpp`, `c_api_test_utils.hpp`
- Test models are stored in `src/test/` subdirectories (e.g., `dummy/`, `passthrough/`, `summator/`)
- Specialized test areas: `src/test/llm/`, `src/test/mediapipe/`, `src/test/python/`, `src/test/embeddings/`

