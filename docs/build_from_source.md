# Building from source

This document gives information how to build docker images and the binary package from source with different variants.

## Prerequisites

1. [Docker Engine](https://docs.docker.com/engine/)
1. Ubuntu 22.04, Ubuntu 24.04 or RedHat 8.10 host
1. make
1. bash

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
- `ubuntu24` for Ubuntu 22.04
- `redhat` for Red Hat UBI 8.10

```bash
make release_image BASE_OS=redhat
```

<hr />

Example:

### `INSTALL_DRIVER_VERSION`

Parameter used to control which GPU driver version will be installed. Supported versions:
| OS | Versions |
|---|---|
| Ubuntu22 | 24.39.31294 (default), <br >24.26.30049 (default), <br />23.22.26516, <br />23.13.2603|
| Ubuntu24 | 24.39.31294 (default) |
| RedHat | 23.22.26516 (default), <br /> 24.26.30049, <br />23.22.26516, <br />22.10.22597 |

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

### `NVIDIA`

By default set to `0`. When set to `1`, there will be additional docker image prepared: `openvino/model_server:latest-cuda` which contains environment required to run inference on NVIDIA GPUs. Please note that such image is significantly larger than the base one.

Hint: optionally use `OV_SOURCE_BRANCH` parameter to specify which branch from [OpenVINO repository](https://github.com/openvinotoolkit/openvino) should be used
and `OV_CONTRIB_BRANCH` to choose the branch from [OpenVINO contrib](https://github.com/openvinotoolkit/openvino_contrib) repository for NVIDIA plugin.

Example:
```bash
make release_image NVIDIA=1 OV_USE_BINARY=0
```

 > **Note**: In order to build the image with redhat UBI8.8 as the base os, it is required to use a host with RedHat subscription and entitlements in `/etc/pki/entitlement` and `/etc/rhsm`. 
That is required to install several building dependencies.

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

When set to `1`, OpenVINO&trade Model Server will be built with the drivers required by [GPU plugin](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) support. Default value: `0`.

Example:
```bash
make release_image GPU=1
```

## Building minimal image

Building minimalistic OVMS docker image requires disabling all optional features:

```bash
make release_image GPU=0 MEDIAPIPE_DISABLE=1 PYTHON_DISABLE=1
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


Read more detailed usage in [developer guide](https://github.com/openvinotoolkit/model_server/blob/main/docs/developer_guide.md).

## Building ovms.exe on Windows

[Developer Guide for Windows](windows_developer_guide.md) document describes experimental/alpha windows deployment for ovms.exe binary.
Follow the instructions to start the ovms server as a standalone binary on a Windows 11 system.
