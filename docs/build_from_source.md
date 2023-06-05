# Building from source

This document gives information how to build docker images and the binary package from source with different variants.

## Prerequisites

1. [Docker Engine](https://docs.docker.com/engine/)
1. Ubuntu 20.04, Ubuntu 22.04 or RedHat 8.7 host
1. make
1. bash

## Makefile and building

Makefile located in root directory of this repository contains all targets needed to build docker images and binary packages.

It contains `docker_build` target which by default builds multiple docker images:
- `openvino/model_server:latest` - smallest release image containing only necessary files to run model server on CPU
- `openvino/model_server:latest-gpu` - release image containing support for Intel GPU and CPU
- `openvino/model_server:latest-nginx-mtls` - release image containing exemplary NGINX MTLS configuration
- `openvino/model_server-build:latest` - image with builder environment containing all the tools to build OVMS

The `docker_build` target also prepares binary package to run OVMS as standalone application and shared library to link against user written C/C++ applications.

```bash
git clone https://github.com/openvinotoolkit/model_server
cd model_server
```
```bash
make docker_build
tree dist/ubuntu
````

```bash
dist/ubuntu
├── Dockerfile.redhat
├── Dockerfile.ubuntu
├── drivers
├── LICENSE
├── Makefile
├── ovms.tar.gz
├── ovms.tar.gz.metadata.json
├── ovms.tar.gz.sha256
├── ovms.tar.xz
├── ovms.tar.xz.metadata.json
├── ovms.tar.xz.sha256
└── thirdparty-licenses
```

## Building Options

### `BASE_OS`

Select base OS:
- `ubuntu` for Ubuntu 20.04 (default)
- `redhat` for Red Hat UBI 8.7

```bash
make docker_build BASE_OS=redhat
```

### `BASE_OS_TAG_UBUNTU`

Select ubuntu base image version:
- `20.04` ubuntu:20.04 (default value)
- `22.04` ubuntu:22.04

```bash
make docker_build BASE_OS_TAG_UBUNTU=22.04
```

<hr />

Example:

### `INSTALL_DRIVER_VERSION`

Parameter used to control which GPU driver version will be installed. Supported versions:
| OS | Versions |
|---|---|
| Ubuntu22 | 23.13.26032 (default), <br />22.35.24055, <br />22.10.22597, <br />21.48.21782 |
| Ubuntu20 | 22.43.24595 (default), <br />22.35.24055, <br />22.10.22597, <br />21.48.21782 |
| RedHat | 22.43.24595 (default), <br />22.28.23726, <br />22.10.22597, <br />21.38.21026 |

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
and run make docker_build with parameter: INSTALL_DRIVER_VERSION=dg2.

Example:
```bash
make docker_build BASE_OS=ubuntu INSTALL_DRIVER_VERSION=dg2
```

<hr />

### `DLDT_PACKAGE_URL`

Parameter used to specify URL to OpenVINO package. By default set to latest release.

<hr />

### `NVIDIA`

By default set to `0`. When set to `1`, there will be additional docker image prepared: `openvino/model_server:latest-cuda` which contains environment required to run inference on NVIDIA GPUs. Please note that such image is significantly larger than the base one.

Hint: use together with `OV_USE_BINARY=0` to force building OpenVINO from source. Use `OV_SOURCE_BRANCH` parameter to specify which branch from [OpenVINO repository](https://github.com/openvinotoolkit/openvino) should be used.
Use together with `OV_CONTRIB_BRANCH` to specify which branch from [OpenVINO contrib](https://github.com/openvinotoolkit/openvino_contrib) repository should be used for NVIDIA plugin.

Example:
```bash
make docker_build NVIDIA=1 OV_USE_BINARY=0 OV_SOURCE_BRANCH=master OV_CONTRIB_BRANCH=master
```

Note. In order to build the image with redhat UBI8.7 as the base os, it is required to use a host with RedHat subscription and entitlements in `/etc/pki/entitlement` and `/etc/rhsm`. 
That is required to install several building dependencies.

<hr />

### `OV_USE_BINARY`

By default set to `1`. When set to `0`, OpenVINO will be built from sources and `DLDT_PACKAGE_URL` will be omitted.  
Use `OV_SOURCE_BRANCH` and `OV_SOURCE_ORG` to select [OpenVINO repository](https://github.com/openvinotoolkit/openvino) branch and fork. By default `master` will be used and org `openvinotoolkit`.  

Example:
```bash
make docker_build OV_USE_BINARY=0 OV_SOURCE_BRANCH=<commit or branch> OV_SOURCE_ORG=<fork org>
```

### `RUN_TESTS`

Enables or disabled unit tests execution as part of the docker image building.
- `0` Unit tests are skipped
- `1` Unit tests are executed (default)

```bash
make docker_build RUN_TESTS=0
```

Running the unit tests will make the building last longer and it will consume a bit more RAM

### `CHECK_COVERAGE`

Enables or disabled calculating the unit tests coverage as part of the docker image building.
- `0` Checking the coverage is skipped
- `1` Checking the coverage is included

```bash
make docker_build RUN_TESTS=0
```

Running the unit tests will increase build time and consume more RAM

### `JOBS`

Number of compilation jobs. By default it is set to the number of CPU cores. On hosts with low RAM, this value can be reduced to avoid out of memory errors during the compilation.

```bash
make docker_build JOBS=2
```
<hr />

### `MEDIAPIPE_DISABLE`

When set to `0`, OpenVINO&trade Model Server will be built with [MediaPipe](mediapipe.md) support. Default value: `0`.

Example:
```bash
make docker_build MEDIAPIPE_DISABLE=0
```

Read more detailed usage in [developer guide](https://github.com/openvinotoolkit/model_server/blob/develop/docs/developer_guide.md).
