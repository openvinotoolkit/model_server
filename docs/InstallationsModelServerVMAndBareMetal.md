# Landing OpenVINO&trade; Model Server on Bare Metal Hosts and Virtual Machines

> **NOTES**:
> * These steps apply to Ubuntu*, CentOS* and RedHat*
> * An internet connection is required to follow the steps in this guide.

## Introduction
OpenVINO&trade; Model Server is a Python* implementation of gRPC and RESTful API interfaces defined by Tensorflow serving. In the backend it uses Inference Engine libraries from OpenVINO&trade; toolkit, which speeds up the execution on CPU, and enables it on FPGA and Movidius devices.

OpenVINO&trade; Model Server can be hosted on a bare metal server, virtual machine or inside a docker container. It is also suitable for landing in Kubernetes environment.

## System Requirements

#### Operating Systems 

* Ubuntu 18.04.x or higher, long-term support (LTS), 64-bit
* CentOS 7.4, 64-bit (for target only)
* Red Hat Enterprise Linux
* ClearLinux
* SUSE Linux

## Overview
This guide provides step-by-step instructions to install OpenVino&trade; Model Server on Virtual Machines or Bare Metal Hosts. Links are provided for different compatible hardwares. Following instructions are covered in this:
- <a href="#model-server-installation">Model Server Installation</a>
- <a href="#model-server-installation">Starting the Serving</a>
- <a href="#using-ncs2">Using Neural Compute Sticks</a>
- <a href="#using-hddl">Using HDDL accelerators</a>
- <a href="#using-igpu">Using iGPU accelerators</a>
- <a href="#using-plugin">Using Multi-Device and Heterogeneous Plugin</a>

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
   make docker_build DLDT_PACKAGE_URL=<URL>
   ````
   The `URL` above represents a link to the OpenVINO Toolkit package that you can get after  registration on [OpenVINO&trade; Toolkit website](https://software.intel.com/en-us/openvino-toolkit/choose-download). The binary package will be located in the docker image `openvino/model_server-build:latest` in the folder `./dist`.

4. The `make docker_build` target will also make a copy of the binary package in a dist subfolder in the model server root directory.

5. Navigate to the folder containing binary package and unpack the included tar.gz file
   ```Bash
   cd dist/centos && tar -xzvf ovms.tar.gz
   ```

## Start the Serving<a name="start-the-serving"></a>
1. The server can be started using the command : 
```Bash
/ovms/bin/ovms -help
```
2. The server can be started in interactive mode, as  a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements.

Refer to [Installing OpenVINO&trade; Model Server for Linux using Docker Container](./InstallationsLinuxDocker.md) to get more details about the docker image built in <a href="#model-server-installation">Model Server Installation</a>

While preparing environment you can use steps from this [Dockerfile](https://github.com/openvinotoolkit/model_server/blob/main/Dockerfile.centos) as a reference.

## Using Neural Compute Sticks 2<a name="using-ncs2"></a>

* OpenVINO Model Server can employ AI accelerators [Intel® Neural Compute Stick and Intel® Neural Compute Stick 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html).

* To use Movidus Neural Compute Sticks with OpenVINO Model Server you need to have OpenVINO Toolkit with Movidius VPU support installed, you need to enable them performing [additional steps for NCS](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html).

* On server startup, you need to specify that you want to load model on Neural Compute Stick for inference execution. You can do that by setting `target_device` parameter to `MYRIAD`. If it's not specified, OpenVINO will try to load model on CPU.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```
You can also run it in [Docker container](./InstallationsLinuxDocker.md)

>Note:
>* Checkout supported configurations. Look at [VPU Plugins documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_VPU.html) to see if your model is supported. If not, take a look at [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) and convert your model to desired format.
>* A single stick can handle one model at a time. If there are multiple sticks plugged in, OpenVINO Toolkit chooses to which one the model is loaded.

## Using HDDL accelerators<a name="using-hddl"></a>
- OpenVINO Model Server can employ High-Density Deep Learning (HDDL) accelerators based on [Intel Movidius Myriad VPUs](https://www.intel.ai/intel-movidius-myriad-vpus/#gs.xrw7cj). To use HDDL accelerators with OpenVINO&trade; Model Server perform [additional steps for Intel® Movidius™ VPU](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html).

- Check status of `hddldaemon`, it should be running and `/dev/ion` character device is present on the host.

- On server startup, you need to specify that you want to load model on HDDL card for inference execution. You can do that by setting `target_device` parameter to `HDDL`. If it's not specified, OpenVINO will try to load model on CPU.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```
You can also run it in [Docker container](./InstallationsLinuxDocker.md)

>Note: Check out supported configurations. Look at VPU Plugins to see if your model is supported. If not, take a look at OpenVINO Model Optimizer and convert your model to desired format.

## Using iGPU accelerators<a name="using-igpu"></a>
- OpenVINO Model Server can use Intel GPU to run inference operations via a [GPU plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CL_DNN.html)
Beside the OVMS binary package, you need to run [additional steps for Intel® GPU](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps).


- On the OVMS server startup, specify that you want to load model on
the GPU for inference execution. Set `target_device` parameter to `GPU`.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

You can also run it in [Docker container](./InstallationsLinuxDocker.md)

>**Note**: Check out [supported configurations](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html).
Look at VPU Plugins to see if your model is supported and use [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) 
and convert your model to the OpenVINO format.

## Using Multi-Device and Heterogeneous Plugin<a name="using-plugin"></a>

See [Multi-Device Plugin overview](./InstallationsLinuxDocker.md#multiplugin) and [Hetero Plugin overview](./InstallationsLinuxDocker.md#heteroplugin).<br>

Apply instructions linked above without Docker specific steps.

