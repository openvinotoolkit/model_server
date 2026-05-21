---
applyTo: "**/{BUILD,BUILD.bazel}"
---
# Bazel BUILD File Guidelines (OVMS)

## Target Granularity

- **One logical unit per `ovms_cc_library` target.** Each `.cpp` file should belong to its own target with its associated `.hpp` in `hdrs`.
- **Do not add sources to monolithic targets.** If you're tempted to add a file to an existing large target, create a new focused target instead.

## Macros

- Use `ovms_cc_library` for production code (sets standard copts, linkopts, local_defines).
- Use `ovms_test_cc_library` for test utility libraries (adds gtest dep automatically).
- Use `ovms_cc_test` for test binaries.
- Use `additional_copts` for feature-flag copts (`COPTS_MEDIAPIPE`, `COPTS_PYTHON`). Do not override `copts` directly.

## Dependencies

- **Include-what-you-use for deps**: list only direct dependencies. Do not rely on transitive deps.
- **Prefer narrow deps**: depend on interface targets (`servable_metadata_provider`, `model_instance_provider`) over heavy implementation targets (`modelmanager`) when possible.
- **Forward-declaration targets**: for lightweight `_fwd.hpp` headers, create a separate header-only target (e.g., `tensorinfo_fwd`) so consumers can depend on it without pulling the full implementation.
- **Use `select()` for optional deps**: MediaPipe, Python, Drogon features should be behind `select()` on the appropriate config setting.

## Visibility

- Default to no `visibility` (package-private). Only add `visibility = ["//visibility:public"]` when the target is genuinely needed outside its package.
- Tight visibility prevents accidental coupling between subsystems.

## Naming Conventions

- Target names: lowercase with descriptive names matching the source (e.g., `grpcservermodule`, `kfs_grpc_frontend`).
- Test targets: match the test file name pattern (e.g., `ensemble_tests` for `ensemble_tests.cpp`).
