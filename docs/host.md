# Bare Metal and Virtual Hosts {#ovms_docs_baremetal}

OpenVINO™ Model Server includes a C++ implementation of gRPC and RESTful API interfaces defined by TensorFlow Serving. 
In the backend, it uses OpenVINO&trade; Runtime libraries from OpenVINO&trade; toolkit, which speeds up the execution on CPU, and enables it on iGPU and Movidius devices.

OpenVINO Model Server can be hosted on a bare metal server, virtual machine, or inside a docker container. It is also suitable for landing in the Kubernetes environment.

**Before you start:**

OpenVINO Model Server execution on baremetal is tested on Ubuntu 20.04.x. For other operating systems we recommend using [OVMS docker containers](./docker_container.md).

For supported hardware, refer to [supported configurations](https://docs.openvino.ai/2022.1/_docs_IE_DG_supported_plugins_Supported_Devices.html).   
Always verify if your model is supported by the VPU Plugins and convert it to the OpenVINO format, using [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).

## Installing Model Server <a name="model-server-installation"></a>

- Clone model server git repository.
- Navigate to the model server directory.
- To install Model Server, you can either use a precompiled binary or build it on your own, in a Docker container.
- Navigate to the folder containing the binary package and unpack the included `tar.gz` file.

Here is an example of this process:

```Bash

git clone https://github.com/openvinotoolkit/model_server

cd model_server   
   
# automatically build a container from source
# it will also place a copy of the binary package in the `dist` subfolder in the Model Server root directory
make docker_build

# unpack the `tar.gz` file
cd dist/ubuntu && tar -xzvf ovms.tar.gz

```

## Running the Server

The server can be started in two ways:

- using the ```./ovms/bin/ovms --help``` command in the folder, where OVMS was is installed
- in the interactive mode - as a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements

Refer to [Running Model Server using Docker Container](./docker_container.md) to get more details on the OpenVINO Model Server parameters and configuration.

> **NOTE**:
> When AI accelerators are used for inference execution, additional steps may be required to install their drivers and dependencies. Learn more about it 
> Learn more about it on [OpenVINO installation guide](https://docs.openvino.ai/2022.1/openvino_docs_install_guides_installing_openvino_linux.html).
