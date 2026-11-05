# Building from source {#ovms_docs_build_from_source}

This document gives information how to build docker images and the binary package from source with different variants.

## Prerequisites

1. [Docker Engine](https://docs.docker.com/engine/) with Buildx CLI plugin or Podman.
1. Host with x86_64 architecture
1. make
1. bash

 > **Note**: Building Windows Model Server is covered in [Developer Guide for Windows](windows_developer_guide.md).

## Makefile and building

Makefile located in root directory of this repository contains all targets needed to build docker images and binary packages. One of these targets is `release_image` which by default builds `openvino/model_server:latest` image.

```bash
git clone https://github.com/openvinotoolkit/model_server
cd model_server
make release_image GPU=1 NPU=1
````

To build the image with non default configuration, add parameters described below.

## Building Options

### `BASE_OS`

Select base OS:
- `ubuntu22` for Ubuntu 22.04
- `ubuntu24` for Ubuntu 24.04 (default)
- `ubuntu26` for Ubuntu 26.04
- `redhat` for Red Hat UBI 9.8

```text
make release_image BASE_OS=redhat
```

<hr />

Example:

### `INSTALL_DRIVER_VERSION`

`INSTALL_DRIVER_VERSION` selects the GPU driver version. The Makefile defaults are:

| Base OS | Default version |
|---|---|
| Ubuntu 22.04 | `24.39.31294` |
| Ubuntu 24.04 | `26.31.39395` |
| Ubuntu 26.04 | `26.31.39395` |
| Red Hat | `25.18.33578` |

See [install_ubuntu_gpu_drivers.sh](../install_ubuntu_gpu_drivers.sh) and [install_redhat_gpu_drivers.sh](../install_redhat_gpu_drivers.sh) for the installer cases.


Example:
```text
make release_image BASE_OS=ubuntu INSTALL_DRIVER_VERSION=26.31.39395 GPU=1
```

<hr />

### `DLDT_PACKAGE_URL`

Parameter used to specify URL to the OpenVINO tar.gz archive, appropriate for the target OS. Here are the [latest public packages from master branch](https://storage.openvinotoolkit.org/repositories/openvino/packages/master/).
Use this parameter together with `OV_USE_BINARY=1`.

<hr />

### `OV_USE_BINARY`

By default set to `0`. With that setting, OpenVINO backend will be built from sources and `DLDT_PACKAGE_URL` will be omitted.
Use `OV_BRANCH` and `OV_REPO` to select the [OpenVINO repository](https://github.com/openvinotoolkit/openvino) branch and repository. Repository parameters use the full GitHub `owner/repository` name without a URL prefix or `.git` suffix, allowing forks with custom repository names. The defaults are `OV_REPO=openvinotoolkit/openvino`, `OV_TOKENIZERS_REPO=openvinotoolkit/openvino_tokenizers`, and `OV_GENAI_REPO=openvinotoolkit/openvino.genai`. Use `OV_TOKENIZERS_BRANCH` and `OV_GENAI_BRANCH` to select the corresponding revisions. By default the tested commits from `versions.mk` are used.
When `OV_USE_BINARY=1`, the OpenVINO backend will be installed from the binary archive set in `DLDT_PACKAGE_URL`.

Example:
```text
make release_image OV_USE_BINARY=0 OV_BRANCH=master OV_REPO=my-org/custom-openvino OV_TOKENIZERS_REPO=my-org/custom-tokenizers OV_GENAI_REPO=my-org/custom-genai
```

Running the unit tests will increase build time and consume more RAM

### `JOBS`

Number of compilation jobs. By default it is set to the number of CPU cores. On hosts with low RAM, this value can be reduced to avoid out of memory errors during the compilation.

```text
make release_image JOBS=2
```
<hr />

### `MEDIAPIPE_DISABLE`

When set to `0`, OpenVINO&trade Model Server will be built with [MediaPipe](mediapipe.md) support. Default value: `0`.

Example:
```text
make release_image MEDIAPIPE_DISABLE=1
```

### `GPU`

When set to `1`, OpenVINO&trade Model Server will be built with the drivers required by [GPU plugin](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) support. Default value: `0`.

Example:
```text
make release_image GPU=1
```

### `TOKEN`

GitHub Personal Access Token (PAT) used to run builds with an authenticated GitHub user when cloning source repositories. This serves as a workaround for GitHub connection stability issues and API rate limits during the build.

Example:
```text
make release_image TOKEN=<your_github_token>
```

## Building minimal image

To omit optional GPU drivers and MediaPipe support. Python runtime artifacts may still be packaged, but Python nodes and MediaPipe-based LLM endpoints are unavailable in this build only classic models serving:

```bash
make release_image GPU=0 MEDIAPIPE_DISABLE=1
```

## Building Binary Package

The binary packages includes the `ovms` executable and linked libraries for bare metal deployments. It includes also a shared library for the model server CAPI interface. Building `ovms.tar.gz` package is possible by using `targz_package` target:

```text
make targz_package
tree dist/ubuntu24
```

```text
dist/ubuntu24
├── ovms.tar.gz
└── ovms.tar.gz.sha256
```

### Optional `espeak-ng` for Speech Generation

`espeak-ng` is optional in OVMS builds. It is used by Kokoro speech generation for:

- non-English grapheme-to-phoneme conversion,
- fallback phonemization of some out-of-vocabulary (OOV) English words.

If you do not need Kokoro non-English support and can accept reduced English OOV fallback behavior, you can:

- skip building `espeak-ng` during image/package build:

```text
make targz_package ESPEAK=0
```

- or remove it from an already prepared package by deleting:
     - `libespeak-ng.so*` from the package `lib` directory
     - `share/espeak-ng-data/`

Consequences of removing or disabling `espeak-ng`:

- Speech generation endpoint still works for Kokoro English usage.
- Non-English Kokoro text normalization/phonemization paths that depend on `espeak-ng` are not available.
- English OOV words no longer use `espeak-ng` fallback, so pronunciation quality/coverage for uncommon words may degrade.
- In practice, treat Kokoro as English-focused (for example `en-us` and `en-gb`) when `espeak-ng` is not present.

---

Read more details about building and testing changes in [developer guide](./developer_guide.md).

