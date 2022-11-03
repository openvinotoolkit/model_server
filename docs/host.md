# Starting Model Server Locally {#ovms_docs_baremetal}

Before starting the server, make sure your hardware is in the list of [supported configurations](https://docs.openvino.ai/2022.2/_docs_IE_DG_supported_plugins_Supported_Devices.html).

> **NOTE**: OpenVINO Model Server execution on baremetal is tested on Ubuntu 20.04.x. For other operating systems, starting model server in a [docker container](./docker_container.md) is recommended.
   
## Installing Model Server <a name="model-server-installation"></a>

1. Clone model server git repository.
2. Navigate to the model server directory.
3. Use a precompiled binary or build it in a Docker container-.
4. Navigate to the folder containing the binary package and unpack the `tar.gz` file.

Run the following commands to build a model server Docker image:

```bash

git clone https://github.com/openvinotoolkit/model_server.git

cd model_server   
   
# automatically build a container from source
# it places a copy of the binary package in the `dist` subfolder in the Model Server root directory
make docker_build

# unpack the `tar.gz` file
cd dist/ubuntu && tar -xzvf ovms.tar.gz

```

In the `./dist` directory it will generate: 

- image tagged as openvino/model_server:latest - with CPU, NCS, and HDDL support
- image tagged as openvino/model_server:latest-gpu - with CPU, NCS, HDDL, and iGPU support
- image tagged as openvino/model_server:latest-nginx-mtls - with CPU, NCS, and HDDL support and a reference nginx setup of mTLS integration
- release package (.tar.gz, with ovms binary and necessary libraries)

> **NOTE**: Model Server docker image can be created with ubi8-minimal base image or the default ubuntu20. Model Server with the ubi base image does not support NCS and HDDL accelerators.

## Running the Server

The server can be started in two ways:

- using the ```./ovms/bin/ovms --help``` command in the folder, where OVMS was is installed
- in the interactive mode - as a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements

Refer to [Running Model Server using Docker Container](./docker_container.md) to get more details on the OpenVINO Model Server parameters and configuration. 


> **NOTE**:
> When AI accelerators are used for inference execution, additional steps may be required to install their drivers and dependencies. 
> Learn more about it on [OpenVINO installation guide](https://docs.openvino.ai/2022.2/openvino_docs_install_guides_installing_openvino_linux.html).

## Building an OpenVINO&trade; Model Server Docker Image from Source <a name="sourcecode"></a>

Running the inference operation on GPU requires the ovms process security context account to have correct permissions.
It has to belong to the render group identified by the command:
```
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```
The default account in the docker image is already preconfigured. In case you change the security context, use the following command
to start the ovms container:
```
docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

> **NOTE**: The public docker image includes the OpenCL drivers for GPU in version 21.38.21026.

### Model Server image with DG2 support (Ubuntu 20.04)

Image with DG2 GPU support has not been published. To build the image yourself you need to have DG2 drivers installed on the host and NEO Runtime packages available. 

Put NEO Runtime packages in the catalog `<model_server_dir>/release_files/drivers/dg2` and run `make docker_build` with parameter: `INSTALL_DRIVER_VERSION=dg2`.

Example:
```
make docker_build BASE_OS=ubuntu OVMS_CPP_DOCKER_IMAGE=ovms_dg2 INSTALL_DRIVER_VERSION=dg2
```

## Using Multi-Device Plugin

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. 
With Multi-Device Plugin enabled, inference requests will be load balanced between multiple devices. 
For more detailed information read [OpenVINO Multi-Device plugin documentation](https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_Running_on_multiple_devices.html).

In order to use this feature in OpenVino™ Model Server, following steps are required:

Set target_device for the model in configuration json file to MULTI:DEVICE_1,DEVICE_2 (e.g. MULTI:MYRIAD,CPU, order of the devices defines their priority, so MYRIAD devices will be used first in this example).


Below is exemplary config.json setting up Multi-Device Plugin for resnet model, using Intel® Movidius™ Neural Compute Stick and CPU devices:
```
make docker_build BASE_OS=ubuntu
```

Additionally, you can use the `INSTALL_DRIVER_VERSION` argument command to choose which GPU driver version is used by the produced image. 
If not provided, most recent version is used.

Currently, the following versions are available:
- 21.38.21026 - Redhat
- 21.48.21782 - Ubuntu

Example:
```bash
make docker_build INSTALL_DRIVER_VERSION=21.38.21026
```
If not provided, version 21.38.21026 is used for Redhat and 21.48.21782 is used for Ubuntu.



Always verify if your model is supported by the VPU Plugins and convert it to the OpenVINO format, using [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).
