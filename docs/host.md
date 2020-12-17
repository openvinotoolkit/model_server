# Landing OpenVINO&trade; Model Server on Bare Metal Hosts and Virtual Machines

> **NOTES**:
> * These steps apply to Ubuntu*, CentOS*
> * An internet connection is required to follow the steps in this guide.

## Introduction
OpenVINO&trade; Model Server includes a C++ implementation of gRPC and RESTful API interfaces defined by Tensorflow serving. 
In the backend it uses Inference Engine libraries from OpenVINO&trade; toolkit, which speeds up the execution on CPU, and enables it on iGPU and Movidius devices.

OpenVINO&trade; Model Server can be hosted on a bare metal server, virtual machine or inside a docker container. It is also suitable for landing in Kubernetes environment.

## System Requirements

#### Operating Systems 

We are testing OpenVINO Model Server execution on baremetal on the following OSes: 
* Ubuntu 20.04.x
* CentOS 7.8

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
3. To install Model Server, you could use precompiled version or built it on your own inside a docker container. Build a docker container with automated steps using the command :
   ```Bash
   make docker_build
   ````
4. The `make docker_build` target will also make a copy of the binary package in a dist subfolder in the model server root directory.

5. Navigate to the folder containing binary package and unpack the included tar.gz file using the command :
   ```Bash
   cd dist/centos && tar -xzvf ovms.tar.gz
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



