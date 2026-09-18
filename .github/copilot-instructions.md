# Copilot Instructions for OpenVINO Model Server (OVMS)

> Canonical guidance for all AI agents lives in **[AGENTS.md](../AGENTS.md)** at the repository
> root. VS Code / Copilot reads that file automatically — start there.

`AGENTS.md` is a lean index: it holds the always-on principles (performance-first C++17,
include-what-you-use, style via `make` targets, license headers, security) and points to
task-scoped rule files.

Path-scoped rules auto-attach while you edit matching files and also live under
[.github/instructions/](instructions):

- [cpp-headers.instructions.md](instructions/cpp-headers.instructions.md) — `*.hpp` rules
- [cpp-sources.instructions.md](instructions/cpp-sources.instructions.md) — `*.cpp` rules
- [bazel-build.instructions.md](instructions/bazel-build.instructions.md) — `BUILD` file rules
- [build-workflow.instructions.md](instructions/build-workflow.instructions.md) — build & test workflow
- [code-review.instructions.md](instructions/code-review.instructions.md) — C++ review standards
- [ovms-auto-validate.instructions.md](instructions/ovms-auto-validate.instructions.md) — post-change validation
