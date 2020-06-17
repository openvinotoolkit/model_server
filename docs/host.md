# Landing OpenVINO&trade; Model Server on Bare Metal Hosts and Virtual Machines

## Requirements

OpenVINO&trade; Model Server installation is fully tested on Ubuntu18.04, Centos7 and ClearLinux, however there are no anticipated issues on other 
Linux distributions like RedHat* or SUSE Linux*.

## Model Server Installation
In order to install OVMS on a baremetal host or a Virtual Machine, you need to unpack the binary package in OVMS.
You could use precompiled version or build it on your own inside a docker container. The building steps are automated 
with a command:
```bash
make docker_build
```
The binary package will be in the docker image ovms-build:latest in the folder  ./dist.
Unpack the included tar.gz file and start ovms via: `/ovms/bin/./ovms --help`

The server can be started in interactive mode, as a background process or a daemon initiated by `systemctl/initd` depending
on the Linux distribution and specific hosting requirements.

Refer to [docker_container.md](docker_container.md) to get more details.


## Using Neural Compute Sticks

OpenVINO Model Server can employ AI accelerators [Intel® Neural Compute Stick and Intel® Neural  Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick).

To use Movidus Neural Compute Sticks with OpenVINO Model Server you need to have OpenVINO Toolkit 
with Movidius VPU support installed.
In order to do that follow [OpenVINO installation instruction](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).
Don't forget about [additional steps for NCS](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux#inpage-nav-4-2).

On server startup, you need to specify that you want to load model on Neural Compute 
Stick for inference execution. You can do that by setting `target_device` parameter to `MYRIAD`. If it's not 
specified, OpenVINO will try to load model on CPU.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```

You can also [run it in Docker container](docker_container.md#starting-docker-container-with-ncs)

**Note**: Checkout [supported configurations](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html).
Look at VPU Plugins to see if your model is supported. If not, take a look at [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) 
and convert your model to desired format.


## Using HDDL accelerators

OpenVINO Model Server can employ High-Density Deep Learning (HDDL)
accelerators based on [Intel Movidius Myriad VPUs](https://www.intel.ai/intel-movidius-myriad-vpus/#gs.xrw7cj).
To use HDDL accelerators with OpenVINO Model Server you need to have OpenVINO
 Toolkit 
with Intel® Vision Accelerator Design with Intel® Movidius™ VPU support installed.
In order to do that follow [OpenVINO installation instruction](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).
Don't forget about [additional steps for Intel® Movidius™ VPU](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#install-VPU).

On server startup, you need to specify that you want to load model on
HDDL card for inference execution. You can do that by setting `target_device` parameter to `HDDL`. If it's not 
specified, OpenVINO will try to load model on CPU.

Example:
```
ovms --model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```

Check out our recommendation for [throughput optimization on HDDL](performance_tuning.md#hddl-accelerators)

You can also [run it in Docker container](docker_container.md#starting-docker-container-with-hddl)

**Note**: Check out [supported configurations](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html).
Look at VPU Plugins to see if your model is supported. If not, take a look at [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) 
and convert your model to desired format.


## Using Multi-Device Plugin

See [Multi-Device Plugin overview](docker_container.md#using-multi-device-plugin)
In order to use Multi-Device Plugin on bare host, simply apply instructions linked above without Docker specific steps.
