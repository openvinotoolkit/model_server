# Building from source {#ovms_docs_build_from_source}

This document gives information how to build docker images and the binary package from source with different variants.

## Prerequisites

1. [Docker Engine](https://docs.docker.com/engine/)
1. Ubuntu 22.04, Ubuntu 24.04 or RedHat 9.6 host
1. make
1. bash

 > **Note**: Building Windows Model Server is covered in [Developer Guide for Windows](windows_developer_guide.md).

## Makefile and building

Makefile located in root directory of this repository contains all targets needed to build docker images and binary packages. One of these targets is `release_image` which by default builds `openvino/model_server:latest` image.

```bash
git clone https://github.com/openvinotoolkit/model_server
cd model_server
make release_image
````

To build the image with non default configuration, add parameters described below.

## Building Options

### `BASE_OS`

Select base OS:
- `ubuntu22` for Ubuntu 22.04 (default)
- `ubuntu24` for Ubuntu 24.04
- `redhat` for Red Hat UBI 9.6

```bash
make release_image BASE_OS=redhat
```

<hr />

Example:

### `INSTALL_DRIVER_VERSION`

Parameter used to control which GPU driver version will be installed. Supported versions:
| OS | Versions |
|---|---|
| Ubuntu22 | 24.39.31294 (default), <br /> 24.26.30049 (default), <br /> 23.22.26516|
| Ubuntu24 | 24.52.32224 (default), <br /> 24.39.31294 |
| RedHat | 23.22.26516 (default), <br /> 24.26.30049, <br />23.22.26516, <br /> 22.10.22597 |

Additionally it is possible to specify custom (pre-production) drivers by providing location to NEO Runtime packages on local disk. Contact Intel representative to get the access to the pre-production drivers.
Warning: _Maintained only for Ubuntu base OS._

Put NEO Runtime deb packages in the catalog `<model_server_dir>/release_files/drivers/dg2`. Expected structure is like below:

```
drivers
└── dg2
     ├── intel-igc-core_<version>_amd64.deb
     ├── intel-igc-opencl_<version>_amd64.deb
     ├── intel-level-zero-gpu-dbgsym_<version>_amd64.deb
     ├── intel-level-zero-gpu_<version>_amd64.deb
     ├── intel-opencl-icd-dbgsym_<version>_amd64.deb
     ├── intel-opencl-icd_<version>_amd64.deb
     ├── libigdgmm12_<version>_amd64.deb
     └── libigdgmm12_<version>_amd64.deb
```
and run make release_image with parameter: INSTALL_DRIVER_VERSION=dg2.

Example:
```bash
make release_image BASE_OS=ubuntu INSTALL_DRIVER_VERSION=dg2
```

<hr />

### `DLDT_PACKAGE_URL`

Parameter used to specify URL to the OpenVINO tar.gz archive, appropriate for the target OS. Here are the [latest public packages from master branch](https://storage.openvinotoolkit.org/repositories/openvino/packages/master/).
Use this parameter together with `OV_USE_BINARY=1`.

<hr />

### `OV_USE_BINARY`

By default set to `0`. With that setting, OpenVINO backend will be built from sources and `DLDT_PACKAGE_URL` will be omitted.
Use `OV_SOURCE_BRANCH` and `OV_SOURCE_ORG` to select [OpenVINO repository](https://github.com/openvinotoolkit/openvino) branch and fork. By default the latest tested commit from `master` branch will be used and org `openvinotoolkit`.
When `OV_USE_BINARY=1`, the OpenVINO backend will be installed from the binary archive set in `DLDT_PACKAGE_URL`.

Example:
```bash
make release_image OV_USE_BINARY=0 OV_SOURCE_BRANCH=<commit or branch> OV_SOURCE_ORG=<fork org>
```

Running the unit tests will increase build time and consume more RAM

### `JOBS`

Number of compilation jobs. By default it is set to the number of CPU cores. On hosts with low RAM, this value can be reduced to avoid out of memory errors during the compilation.

```bash
make release_image JOBS=2
```
<hr />

### `PYTHON_DISABLE`

When set to `0`, OpenVINO&trade Model Server will be built with [Python Nodes](python_support/quickstart.md) support. Default value: `0`.

Example:
```bash
make release_image PYTHON_DISABLE=1
```

### `MEDIAPIPE_DISABLE`

When set to `0`, OpenVINO&trade Model Server will be built with [MediaPipe](mediapipe.md) support. Default value: `0`.

Example:
```bash
make release_image MEDIAPIPE_DISABLE=1 PYTHON_DISABLE=1
```

 > **Note**: In order to build the image with python nodes support (PYTHON_DISABLE=0) mediapipe have to be enabled (MEDIAPIPE_DISABLE=0)

### `GPU`

When set to `1`, OpenVINO&trade Model Server will be built with the drivers required by [GPU plugin](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) support. Default value: `0`.

Example:
```bash
make release_image GPU=1
```

### `TARGETARCH`

Target CPU architecture of the produced image. Default value: `amd64`. Set to `arm64`
to build a native `aarch64` image (see [Building for ARM64](#building-for-arm64-aarch64)
below). On `arm64` the build automatically selects the ARM-appropriate toolchain
options (branch protection instead of Intel CET), the `aarch64` OpenVINO runtime
library layout, and skips the x86-only Intel GPU/NPU components.

Example:
```bash
make ovms_builder_image TARGETARCH=arm64 BASE_OS=ubuntu22 OV_USE_BINARY=1 DLDT_PACKAGE_URL=<aarch64 OpenVINO GenAI package>
```

## Building minimal image

Building minimalistic OVMS docker image requires disabling all optional features:

```bash
make release_image GPU=0 MEDIAPIPE_DISABLE=1 PYTHON_DISABLE=1
```

## Building for ARM64 (aarch64)

OpenVINO Model Server can be built and run natively on `aarch64` (ARM64) Linux as a
CPU-only image. The build must run **on an `aarch64` host** (cross-building is not
supported here) and uses pre-built OpenVINO binaries via `OV_USE_BINARY=1`.

Notes specific to ARM64:
- Set `TARGETARCH=arm64`.
- The OpenVINO GenAI nightly packages publish `aarch64` archives for **Ubuntu 22.04
  only**, so build with `BASE_OS=ubuntu22` and point `DLDT_PACKAGE_URL` at the
  matching `..._arm64.tar.gz` archive.
- ARM64 is CPU-only: the Intel GPU/NPU drivers and `intel-opencl-icd` are x86-only
  and are skipped automatically. Build with `GPU=0` and do not request the GPU image
  variant.
- `MEDIAPIPE_DISABLE=1 PYTHON_DISABLE=1` are recommended for a minimal image.

Build the release image:
```bash
make release_image \
    TARGETARCH=arm64 \
    BASE_OS=ubuntu22 \
    OV_USE_BINARY=1 \
    DLDT_PACKAGE_URL=https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/<version>/openvino_genai_ubuntu22_<version>_arm64.tar.gz \
    GPU=0 MEDIAPIPE_DISABLE=1 PYTHON_DISABLE=1
```

Verify the resulting image runs and serves a model:
```bash
docker run --rm openvino/model_server:latest --version
docker run -d --name ovms -p 8000:8000 \
    -v $(pwd)/src/test/dummy:/models/dummy:ro \
    openvino/model_server:latest \
    --model_name dummy --model_path /models/dummy --rest_port 8000
curl http://localhost:8000/v2/health/ready
```

## Building Binary Package

The binary packages includes the `ovms` executable and linked libraries for bare metal deployments. It includes also a shared library for the model server CAPI interface. Building `ovms.tar.gz` package is possible by using `targz_package` target:

```bash
make targz_package PYTHON_DISABLE=1
tree dist/ubuntu22
````

```bash
dist/ubuntu22
├── ovms.tar.gz
└── ovms.tar.gz.sha256
```

### Optional `espeak-ng` for Speech Generation

`espeak-ng` is optional in OVMS builds. It is used by Kokoro speech generation for:

- non-English grapheme-to-phoneme conversion,
- fallback phonemization of some out-of-vocabulary (OOV) English words.

If you do not need Kokoro non-English support and can accept reduced English OOV fallback behavior, you can:

- skip building `espeak-ng` during image/package build:

```bash
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

