# Building from source

This document gives information how to build docker images and the binary package from source with different variants.

## Prerequisites

1. [Docker Engine](https://docs.docker.com/engine/)
1. Ubuntu 20.04 or RedHat 8.7 host
1. make
1. bash

## Makefile and building

Makefile located in root directory of this repository contains all targets needed to build docker images and binary packages.

It contains `docker_build` target which by default builds multiple docker images:
- `openvino/model_server:latest` - smallest release image containing only neccessary files to run model server on CPU, NCS and HDDL
- `openvino/model_server:latest-gpu` - release image containing support for Intel GPU
- `openvino/model_server:latest-nginx-mtls` - release image containing examplary NGINX MTLS configuration
- `openvino/model_server-build:latest` - image with builder environment containing all the tools to build OVMS

The `docker_build` target also prepares binary package to run OVMS as standalone application and shared library to link against user written C/C++ applications.

```bash
git clone https://github.com/openvinotoolkit/model_server
cd model_server
make docker_build
tree dist/ubuntu
````

```
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

### `INSTALL_DRIVER_VERSION`

Parameter used to control which GPU driver version will be installed. Additionally it is possible to specify custom (pre-production) drivers by providing location to NEO Runtime packages on local disk. Contact Intel representative to get the access to the pre-production drivers.

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
```
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
make docker_build NVIDIA=1 OV_USE_BINARY=0 OV_SOURCE_BRANCH=releases/2022/3 OV_CONTRIB_BRANCH=releases/2022/3
```
```bash
docker run -it --gpus all -p 9178:9178 -v ${PWD}/models/public/resnet-50-tf:/opt/model openvino/model_server:latest-cuda --model_path /opt/model --model_name resnet --target_device NVIDIA
```

Read more detailed usage in [developer guide](b/develop/docs/developer_guide.md).
 