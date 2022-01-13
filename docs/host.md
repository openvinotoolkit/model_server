# Bare Metal and Virtual Machine (VM) Hosts {#ovms_docs_baremetal}

## Introduction
OpenVINO&trade; Model Server includes a C++ implementation of gRPC and RESTful API interfaces compatible with TensorFlow Serving. 
In the backend, Model Server leverages the Inference Engine libraries from OpenVINO&trade; to accelerate inference execution on Intel CPU, iGPU and Movidius VPU devices.

OpenVINO&trade; Model Server can be hosted on a bare metal servers, virtual machines (VMs) or inside containers. The server can also be deployed in Kubernetes and OpenShift clusters.

## System Requirements

#### Operating Systems 

We are testing OpenVINO Model Server execution on baremetal on Ubuntu 20.04.x

For other operating systems we recommend using [OVMS docker containers](./docker_container.md).


#### Hardware 

Check out [supported configurations](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html).

Look at VPU Plugins to see if your model is supported and use [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) and convert your model to the OpenVINO format.



## Model Server Installation<a name="model-server-installation"></a>
1. Clone model server git repository using command :
   ```Bash
   git clone https://github.com/openvinotoolkit/model_server
   ```

2. Navigate to model server directory using command :
   ```Bash
   cd model_server
   ```
3. To install Model Server, it is possible to use a precompiled binary or built it on your own inside a Docker container. To automatically build a container from source, use the following command :
   ```Bash
   make docker_build
   ```
4. Running `make docker_build` will also place a copy of the binary package in the `dist` subfolder in the Model Server root directory.

5. Navigate to the folder containing the binary package and unpack the included tar.gz file using the following command :
   ```Bash
   cd dist/ubuntu && tar -xzvf ovms.tar.gz
   ```

## Running the Serving
1. The server can be started using the command in the folder, where OVMS was installed: 
```Bash
./ovms/bin/ovms --help
```
2. The server can be started in interactive mode, as  a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements.

Refer to [Running Model Server using Docker Container](./docker_container.md) to get more details about the ovms parameters and configuration.


**Note** When AI accelerators are used for inference execution, there might be needed additional steps to install their drivers and dependencies. 
Learn more about it on [OpenVINO installation guide](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html).



