---
name: build-builder-image
description: "Use when building (or rebuilding) the OVMS -build Docker image on the host via the repository Makefile. Covers `make ovms_builder_image`, the BASE_OS variants (ubuntu24, ubuntu22, redhat), proxy forwarding, GPU / NPU / Mediapipe / Python / fuzzer / debug toggles, NO_DOCKER_CACHE / USE_BUILDX flags, and JOBS parallelism. Also covers the full pipeline target `make docker_build` (builder + targz_package + release images). Trigger phrases: 'build the build container', 'build builder image', 'rebuild -build image', 'make ovms_builder_image', 'make docker_build', 'new build image', 'BASE_OS=redhat', 'rebuild docker build image'."
---

# Build the OVMS `-build` Docker Image (Host Make)

Use this skill when the user wants to **(re)build the `*-build` Docker image on the host** using the repository `Makefile`. This is the image other skills (`build-bazel-target`, `run-single-gtest`) execute inside.

Building the `-build` image is **time-expensive** (10s of minutes to over an hour, depending on flags and cache). Always check first whether an existing image can be reused. Only build a new one when the user explicitly requests it, or when dependencies / Dockerfiles / build env changed.

## When to use

- User asks to build, rebuild, or refresh the `-build` Docker image.
- User mentions `make ovms_builder_image`, `make docker_build`, `Dockerfile.ubuntu`, `Dockerfile.redhat`, `BASE_OS`, `NO_DOCKER_CACHE`.
- A previous build failed inside the container with toolchain / dependency errors and a rebuild of the builder image is the documented fix.

## Do NOT use

- For building Bazel targets (`//src:ovms`, `//src:ovms_test`) — use `build-bazel-target`.
- For running tests — use `run-single-gtest`.
- For producing the final release `.tar.gz` only (use `make targz_package`) or release runtime images only (`make ovms_release_images`).

## Pre-flight

1. Confirm the workspace is the OVMS repo root (contains `Makefile`, `Dockerfile.ubuntu`, `Dockerfile.redhat`).
2. Check for an existing `-build` image first — building from scratch is expensive:
   ```bash
   docker images | grep -- -build
   ```
   If a suitable image exists and the user has not asked for a rebuild, **stop here** and tell them to reuse it.
3. Confirm host prerequisites: `docker` available, sufficient free disk (typically ≥ 30 GB), and proxy env vars exported if the network is restricted.
4. **Resolve the image tag (`OVMS_CPP_IMAGE_TAG`) before invoking make.** Do **not** silently fall back to `latest` — it would overwrite the shared default image and make `docker ps` ambiguous about which build belongs to which feature.
   ```bash
   echo "${OVMS_CPP_IMAGE_TAG:-<unset>}"
   ```
   - If `OVMS_CPP_IMAGE_TAG` is **already exported** in the user's shell, reuse that value verbatim and confirm with the user.
   - If it is **unset**, ask the user one of:
     > "`OVMS_CPP_IMAGE_TAG` is not set. What tag should I use? Suggested patterns:
     > - feature/branch tag, e.g. `fix_pull_all_dl_failed`, `lfs-cancel`, `pr-4156`
     > - your username, e.g. `$USER` (→ `${USER}`)
     > Reply with the tag, or explicitly say \"use latest\" if you really want to overwrite `:latest`."
   - Only use `latest` when the user **explicitly** asks for it.
   - Pass the chosen tag on the make command line: `OVMS_CPP_IMAGE_TAG=<tag>`. Optionally also set `IMAGE_TAG_SUFFIX` if the user wants a suffix appended to an existing tag rather than a full override.

## Default invocation

The default builder-image target on Ubuntu 24.04 (after the Pre-flight step has resolved `OVMS_CPP_IMAGE_TAG`):

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag>
```

This produces image `openvino/model_server-build:<resolved_tag>` (tag built from `OVMS_CPP_DOCKER_IMAGE` × `OVMS_CPP_IMAGE_TAG`). Do **not** invoke `make ovms_builder_image` without the tag override unless the user explicitly chose `latest`.

To run the **full** pipeline (builder image → packaged `.tar.gz` → release CPU/GPU images):

```bash
make docker_build OVMS_CPP_IMAGE_TAG=<resolved_tag>
```

## Common Makefile flags

All flags are passed on the `make` command line as `KEY=value` pairs (e.g. `make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag> BASE_OS=redhat NO_DOCKER_CACHE=true`).

| Flag | Default | Purpose |
|------|---------|---------|
| `BASE_OS` | `ubuntu24` | Base distro for the builder image. Supported: `ubuntu24`, `ubuntu22`, `redhat`. |
| `BASE_OS_TAG_UBUNTU` | `24.04` | Override Ubuntu base tag. |
| `BASE_OS_TAG_REDHAT` | `9.7` | Override RHEL/UBI9 base tag. |
| `OV_USE_BINARY` | `1` (Ubuntu) / `0` (RedHat) | Use prebuilt OpenVINO archive vs. build from source. RHEL must use `0`. |
| `DLDT_PACKAGE_URL` | from `versions.mk` | URL of the OpenVINO binary tarball when `OV_USE_BINARY=1`. |
| `MEDIAPIPE_DISABLE` | `0` | Set `1` to disable MediaPipe (also forces `PYTHON_DISABLE=1`). |
| `PYTHON_DISABLE` | `0` | Set `1` to omit Python custom-node bindings. |
| `FUZZER_BUILD` | `0` | Set `1` for fuzzer build (incompatible with `RUN_TESTS=1`, `CHECK_COVERAGE=1`, `BASE_OS=redhat`). |
| `BAZEL_BUILD_TYPE` | `opt` | `dbg` for debug symbols + no strip. |
| `MINITRACE` | `OFF` | `ON` enables minitrace instrumentation. |
| `OV_TRACING_ENABLE` | `0` | `1` enables OpenVINO tracing macros. |
| `RUN_TESTS` | `0` | `1` runs C++ tests inside the build stage (lengthens build). |
| `BUILD_TESTS` | `0` | `1` only compiles tests inside the build stage. |
| `CHECK_COVERAGE` | `0` | `1` enables coverage instrumentation (requires `RUN_TESTS=1`). |
| `RUN_GPU_TESTS` | unset | Set to enable GPU tests during the build. |
| `GPU` / `NPU` | `0` / `0` | Toggle accelerator runtimes in release stage (used by `ovms_release_images`). |
| `JOBS` | `$(CORES_TOTAL)` | Parallelism passed to the build stage; default is derived from physical cores (`sockets × cores-per-socket` via `lscpu`). |
| `NO_DOCKER_CACHE` | unset | `true` adds `--no-cache` and re-pulls the base image. |
| `USE_BUILDX` | unset | `true` switches to `docker buildx build` (BuildKit). |
| `IMAGE_TAG_SUFFIX` | empty | Appended to the image tag (e.g. `-myfeature`). |
| `OVMS_CPP_DOCKER_IMAGE` | `openvino/model_server` | Override image repo. |
| `OVMS_CPP_IMAGE_TAG` | `latest` | Override image tag. **Always set explicitly** — see Pre-flight step 4. |
| `OVMS_METADATA_FILE` | unset | Path to a JSON file copied into the image as `.workspace/metadata.json`. |
| `BUILD_NGINX` | `0` | `1` enables nginx integration. |
| `OV_SOURCE_BRANCH` / `OV_SOURCE_ORG` | from `versions.mk` | Override OpenVINO sources when building OV from source. |
| `OV_GENAI_BRANCH` / `OV_GENAI_ORG` | from `versions.mk` | Override OpenVINO GenAI sources. |
| `OV_TOKENIZERS_BRANCH` / `OV_TOKENIZERS_ORG` | from `versions.mk` | Override OV tokenizers sources. |

The `BUILD_ARGS` variable in the Makefile consolidates these into `--build-arg` flags passed to the underlying `docker build` invocation; you do not need to construct them manually.

## Recipes

### 1. Default Ubuntu 24 builder, reuse cache

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag>
```

### 2. Red Hat (UBI9) builder

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag> BASE_OS=redhat
```

The Makefile auto-selects `--//:distro=redhat` for Bazel inside the image and forces `OV_USE_BINARY=0`.

### 3. Force a clean rebuild (no Docker cache)

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag> NO_DOCKER_CACHE=true
```

### 4. Debug build with extra parallelism

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag> BAZEL_BUILD_TYPE=dbg JOBS=32
```

### 5. Tag a side-by-side image for a feature branch

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag> IMAGE_TAG_SUFFIX=-fix_pull_all_dl_failed
# produces openvino/model_server-build:<resolved_tag>-fix_pull_all_dl_failed
```

### 6. Slim build without Mediapipe / Python custom nodes

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag> MEDIAPIPE_DISABLE=1 PYTHON_DISABLE=1
```

### 7. Full pipeline (builder → tar.gz → release images)

```bash
make docker_build OVMS_CPP_IMAGE_TAG=<resolved_tag> BASE_OS=ubuntu24
# or
make docker_build OVMS_CPP_IMAGE_TAG=<resolved_tag> BASE_OS=redhat
```

### 8. Use BuildKit / buildx

```bash
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag> USE_BUILDX=true
```

## Proxy forwarding

The Makefile automatically forwards `http_proxy`, `https_proxy`, and `no_proxy` from the host environment into `docker build` via `--build-arg`. **Export them in your shell before running make** if your network requires a proxy:

```bash
export http_proxy=http://proxy:port
export https_proxy=$http_proxy
export no_proxy=localhost,127.0.0.1,.intel.com
make ovms_builder_image OVMS_CPP_IMAGE_TAG=<resolved_tag>
```

## Validate after build

```bash
docker images | grep openvino/model_server-build
```

Expect a fresh `openvino/model_server-build:<tag>` image. To smoke-test it, start a container and run a trivial Bazel query:

```bash
docker run --rm \
  -v "$(pwd)":/ovms \
  -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" -e no_proxy="$no_proxy" \
  openvino/model_server-build:${OVMS_CPP_IMAGE_TAG} \
  bash -lc 'cd /ovms && bazel info release'
```

## Tips & pitfalls

- **Never silently use the default `latest` tag.** Resolve `OVMS_CPP_IMAGE_TAG` from the user's environment or ask explicitly (Pre-flight step 4). Overwriting `:latest` clobbers the shared default and makes `docker ps` / `docker images` ambiguous about which build belongs to which feature.
- **Reuse before rebuild.** Always `docker images | grep -- -build` first; recreate only when dependencies, `Dockerfile.*`, `WORKSPACE`, `third_party/**`, or `versions.mk` actually changed.
- **`NO_DOCKER_CACHE=true` is expensive.** Use it only when you suspect cache poisoning, base-image drift, or a broken layer.
- **Distro/`OV_USE_BINARY` constraints are enforced** by the Makefile. RedHat requires `OV_USE_BINARY=0`; Ubuntu 22 requires `OV_USE_BINARY=1`.
- **`MEDIAPIPE_DISABLE=1` requires `PYTHON_DISABLE=1`** — the Makefile errors out otherwise.
- **`FUZZER_BUILD=1` is incompatible** with `RUN_TESTS=1`, `CHECK_COVERAGE=1`, and `BASE_OS=redhat`.
- **Image tag collisions.** Without `IMAGE_TAG_SUFFIX`, repeated builds overwrite `:latest`. Use a suffix when keeping multiple branches' images side by side.
- **Disk space.** Multi-stage builds cache large layers; clean up periodically with `docker image prune` and `docker builder prune` (ask the user before pruning).
- **Long-running operation.** Run as an async terminal so the conversation is not blocked, and poll output with `get_terminal_output`. Never sleep/poll inside the shell.
- **Do not run `bazel` or `make ovms_builder_image` inside the `-build` container.** This skill is host-side only.
