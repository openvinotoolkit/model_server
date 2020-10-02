# Landing OpenVINO&trade; Model Server on Bare Metal Hosts and Virtual Machines

## Requirements

OpenVINO&trade; Model Server installation is fully tested on Ubuntu18.04 and Centos7, however there are no anticipated issues on other 
Linux distributions like ClearLinux*, RedHat* or SUSE Linux*.

## Model Server Installation
In order to install OVMS on a baremetal host or a Virtual Machine, you need to unpack the binary package in OVMS.
You can build it inside a docker container. The building steps are automated 
with a command:
```bash
make docker_build DLDT_PACKAGE_URL=<URL>
```
The `URL` above represents a link to the OpenVINO Toolkit package that you can get after 
registration on [OpenVINO™ Toolkit website](https://software.intel.com/en-us/openvino-toolkit/choose-download).
The binary package will be located in the docker image openvino/model_server-build:latest in the folder  ./dist.
The `make docker_build` target will also make a copy of the binary package in a `dist` subfolder in the model server root directory.
Unpack the included tar.gz file and start ovms via: `/ovms/bin/ovms --help`


The server can be started in interactive mode, as a background process or a daemon initiated by `systemctl/initd` depending
on the Linux distribution and specific hosting requirements.

Refer to [docker_container.md](docker_container.md) to get more details.

While preparing environment you can use steps from this [Dockerfile](../release_files/Dockerfile.centos) as a reference.


## Using Neural Compute Sticks 2

OpenVINO Model Server can employ AI accelerators [Intel® Neural Compute Stick and Intel® Neural  Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick).

To use Movidus Neural Compute Sticks with OpenVINO Model Server, beside unpacking the binary package, you need to enable them by performing 
[additional steps for NCS](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps).

On OVMS server startup, you need to specify that you want to load model on Neural Compute 
Stick for inference execution. You can do that by setting `target_device` parameter to `MYRIAD`. By default, OpenVINO will load model on CPU.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```

You can also [run it in Docker container](docker_container.md#starting-docker-container-with-ncs)

**Note**: Checkout [VPU plugin documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_VPU.html) 
to see if your model is supported. Next, take a look at [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) 
and convert your model to the OpenVINO Intermediate Representation format.


## Using HDDL accelerators

OpenVINO Model Server can employ High-Density Deep Learning (HDDL)
accelerators based on [Intel Movidius Myriad VPUs](https://www.intel.ai/intel-movidius-myriad-vpus/#gs.xrw7cj).
Beside the OVMS binary package, you need to run [additional steps for Intel® Movidius™ VPU](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux_ivad_vpu.html).
Make sure the hddldaemon is running and `/dev/ion` character device is present on the host.

On the OVMS server startup, you need to specify that you want to load model on
HDDL card for inference execution. You can do that by setting `target_device` parameter to `HDDL`. By default, OpenVINO will load model on CPU.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```

You can also [run it in Docker container](docker_container.md#starting-docker-container-with-hddl)

**Note**: Check out [VPU plugin documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_VPU.html)
to see if your model is supported and convert your model to the OpenVINO format via the 
[OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) .

## Using iGPU accelerators

OpenVINO Model Server can use Intel GPU to run inference operations via a [GPU plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CL_DNN.html).
Beside the OVMS binary package, you need to run [additional steps for Intel® GPU](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps).


On the OVMS server startup, specify that you want to load model on
the GPU for inference execution. Set `target_device` parameter to `GPU`.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

You can also [run it in Docker container](docker_container.md#using-gpu-for-inference-execution)

**Note**: Check out [supported configurations](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html).
Look at VPU Plugins to see if your model is supported and use [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) 
and convert your model to the OpenVINO format.

## Using Multi-Device and Heterogeneous Plugin

See also [Multi-Device Plugin and Hetero Plugin overview](docker_container.md#support-for-ai-accelerators).<br>
Apply instructions linked above without Docker specific steps.
