# AGENTS.md — OpenVINO Model Server (OVMS)

Canonical, tool-agnostic guidance for AI coding agents (Copilot, Codex, Claude, Cursor, …).
This file is an **index**: it states the few always-on principles and points to detailed,
task-scoped rule files. Read the linked file when your task matches its scope.

## Project

OVMS is a high-performance inference serving platform built on **OpenVINO** and
**OpenVINO GenAI**. Primarily **C++17**, built with **Bazel**, orchestrated via **Makefile**
and multi-stage **Dockerfiles** (Ubuntu & Red Hat); Windows via `.bat` scripts.

**Performance is a top priority** — both throughput and latency. Evaluate every change for its
performance impact. Avoid copies, allocations, and blocking operations on the hot inference path.

## Repository map

- `src/` — main C++ source (server, gRPC/REST handlers, model management, DAG pipelines,
  MediaPipe, LLM, C API)
- `src/test/` — gtest-based C++ unit tests (`*_test.cpp`); most developer tests live here
- `src/python/` — Python custom node bindings
- `client/` — client libraries (C++, Python, Go, Java)
- `demos/` — end-user demos
- `docs/` — documentation
- `third_party/` — Bazel third-party dependency definitions
- `Dockerfile.ubuntu` / `Dockerfile.redhat` — multi-stage Linux builds
- `Makefile` — Docker-based build/test orchestration
- `*.bat` — Windows build/test scripts

## Always-on principles

- **Performance first** on the hot path — no unnecessary copies, allocations, `dynamic_cast`, or blocking calls.
- **Include-what-you-use**; prefer forward declarations in headers, full includes in `.cpp`. Build times are long — do not add unneeded includes.
- **No dead code**: remove unused functions, variables, includes, and orphan files.
- **Style is enforced** via `make` targets (`make clang-format`, `make cpplint`, `make cppclean`, `make spell`) run on the **host**, not inside the build container. Never invoke `clang-format`/`cpplint` by hand.
- **Apache 2.0 license header** is required on all source files.
- **Security**: follow OWASP Top 10; validate at system boundaries; never log secrets (e.g. API keys).

## Task index — read the matching file for details

| When you are… | Read |
|---|---|
| Editing a C++ **header** (`*.hpp`) | [.github/instructions/cpp-headers.instructions.md](.github/instructions/cpp-headers.instructions.md) |
| Editing a C++ **source** (`*.cpp`) | [.github/instructions/cpp-sources.instructions.md](.github/instructions/cpp-sources.instructions.md) |
| Editing a **BUILD** file | [.github/instructions/bazel-build.instructions.md](.github/instructions/bazel-build.instructions.md) |
| **Building / testing** OVMS | [.github/instructions/build-workflow.instructions.md](.github/instructions/build-workflow.instructions.md) |
| Reviewing or writing C++ to review standards | [.github/instructions/code-review.instructions.md](.github/instructions/code-review.instructions.md) |
| Finishing a C++ change (validate) | [.github/instructions/ovms-auto-validate.instructions.md](.github/instructions/ovms-auto-validate.instructions.md) |

In VS Code / Copilot these files auto-attach via their `applyTo` globs. Other agents should
open the relevant file from the table above on demand.
