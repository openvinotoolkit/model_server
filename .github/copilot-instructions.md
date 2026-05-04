# Copilot Instructions for OpenVINO Model Server (OVMS)

## Project Overview

OpenVINO Model Server (OVMS) is a high-performance inference serving platform built on top of **OpenVINO** and **OpenVINO GenAI**. The codebase is primarily **C++** with **Bazel** as the build system. Supporting infrastructure uses **Makefiles**, **Dockerfiles** (Ubuntu & Red Hat), and **batch files** (Windows).

**Performance is a top priority** — both **throughput** and **latency** are critical. Code changes should be evaluated for their performance impact. Avoid unnecessary copies, allocations, and blocking operations on the hot path.

## Repository Structure

- `src/` — Main C++ source code (server, gRPC/REST handlers, model management, pipelines, mediapipe, LLM, C API)
- `src/test/` — C++ unit tests (gtest-based); this is where most developer tests live
- `src/python/` — Python custom node bindings and related code
- `demos/` — End-user demo applications
- `client/` — Client libraries (C++, Python, Go, Java)
- `docs/` — Documentation
- `third_party/` — Third-party dependency definitions for Bazel
- `Dockerfile.ubuntu` / `Dockerfile.redhat` — Multi-stage Dockerfiles for Linux builds
- `Makefile` — Orchestrates Docker-based builds and test runs

## Code Style

- C++ style is enforced via `cpplint` and `clang-format`
- Run `make style` to check formatting
- Apache 2.0 license headers are required on all source files

## Code Review Instructions for PRs

When analyzing a Pull Request, follow this protocol:

### C++ Guidelines & Performance

1. Follow **C++ Core Guidelines** strictly. Include references in review comments.
2. Check for **hidden performance costs**: avoid `dynamic_cast` on the hot path; suggest `static_cast` or redesign if the type is known.
3. **Avoid copies**: ensure large data structures (tensors, buffers) are passed by reference or moved, not copied.
4. Pass non-fundamental values by `const` reference wherever possible.
5. Use `const` and `constexpr` wherever possible.
6. Prefer member initializer lists over direct assignments in constructor bodies.

### Naming, Structure & Duplication

7. Use descriptive function and variable names. Avoid duplicate code — extract common functionality into reusable utilities.
8. When initial container values are known upfront, prefer initializer-list / brace-initialization over constructing an empty container and inserting.
9. Verify that the result of every newly introduced function is used in at least one call site (except `void` functions).
10. No `using namespace std; using namespace ov;`. Prefer explicit using with specific symbols if needed.
11. No `auto` for primitive types where it obscures readability.

### Includes & Build Time

12. **Unused functions and includes are not allowed.** Build times are already long — do not add unnecessary `#include` directives.
    - **Ban umbrella includes in headers** (`openvino/openvino.hpp`, `rapidjson/document.h`, etc.). Include only the specific subheader you use.
    - **Forward-declare in headers, include in `.cpp`**: if a header only uses pointers or references to a type, use a forward declaration (`class Foo;`) instead of `#include "foo.hpp"`.
    - **Keep headers self-contained but minimal**: each header must compile on its own, but should not pull in transitive dependencies that callers don't need.
    - **Never include a header solely for a typedef or enum**: forward-declare the enum (`enum class Foo;`) or relocate the typedef to a lightweight `fwd.hpp`-style header.
    - **Prefer opaque types / Pimpl**: for complex implementation details, consider the Pimpl idiom to keep implementation-only types out of the public header.
13. **Every new `.cpp` file must belong to its own `ovms_cc_library` target** in a BUILD file. Do not add sources to monolithic targets.
14. **Cross-module coupling goes through interfaces**, not concrete classes. Expose narrow interfaces (`ModelInstanceProvider`, `ServableMetadataProvider`, `ServableNameChecker`) that return plain data and hide lifecycle objects inside implementations.

### Safety & Testing

15. Be mindful when accepting `const T&` in constructors or functions that store the reference: verify that the referenced object's lifetime outlives the usage to avoid dangling references.
16. **Test coverage**: ensure that new features or changes have corresponding tests in `src/test/`.
17. **Documentation**: ensure new public APIs have docstrings in C++ headers and Python bindings; update `docs/` as needed.

