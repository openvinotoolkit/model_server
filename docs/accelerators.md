# Using AI Accelerators {#ovms_docs_target_devices}


## Starting the server with the Intel® Neural Compute Stick 2

[Intel Movidius Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick) can be employed by OVMS OpenVINO Model Server via 
[the MYRIAD plugin](https://docs.openvino.ai/2022.1/openvino_docs_OV_UG_supported_plugins_MYRIAD.html). It must be visible and accessible on the host machine.

NCS devices should be reported by the `lsusb` command, printing out `ID 03e7:2485`.

To start the server with Neural Compute Stick use either of the two options:

1. More securely, without the docker privileged mode and mounting only the usb devices.

   ```bash
   docker run --rm -it -u 0 --device-cgroup-rule='c 189:* rmw' -v /opt/model:/opt/model -v /dev/bus/usb:/dev/bus/usb -p 9001:9001 openvino/model_server \
   --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
   ```

2. less securely, in the docker privileged mode and mounting all devices.
   ```bash
   docker run --rm -it --net=host -u root --privileged -v /opt/model:/opt/model -v /dev:/dev -p 9001:9001 openvino/model_server \
   --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
   ```

## Starting a Docker Container with HDDL

To run a container that is using the HDDL accelerator, _hddldaemon_ must be running on the host machine. 
You must set up the environment (the OpenVINO package must be pre-installed) and start _hddldaemon_ on the host before starting a container. 
Refer to the steps from [OpenVINO installation guides](https://docs.openvino.ai/2022.1/openvino_docs_install_guides_installing_openvino_docker_linux.html#running-the-image-on-intel-vision-accelerator-design-with-intel-movidius-vpus).

An example of a command starting a server with HDDL:
```bash

# --device=/dev/ion:/dev/ion mounts the accelerator device
# -v /var/tmp:/var/tmp enables communication with _hddldaemon_ running on the host machine
docker run --rm -it --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest \

--model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```

Check out our recommendations for [throughput optimization on HDDL](performance_tuning.md).

> **NOTE**:
> the OpenVINO Model Server process within the container communicates with _hddldaemon_ via unix sockets in the `/var/tmp` folder. 
> It requires RW permissions in the docker container security context. 
> It is recommended to start the docker container in the same context as the account starting _hddldaemon_. For example, if you start the _hddldaemon_ as root, add `--user root` to the `docker run` command.

## Starting a Docker Container with Intel GPU

The [GPU plugin](https://docs.openvino.ai/2022.1/openvino_docs_OV_UG_supported_plugins_GPU.html) uses the Intel Compute Library for 
Deep Neural Networks ([clDNN](https://01.org/cldnn)) to infer deep neural networks. For inference execution, it employs Intel® Processor Graphics including 
Intel® HD Graphics, Intel® Iris® Graphics, Intel® Iris® Xe Graphics, and Intel® Iris® Xe MAX graphics.


Before using GPU as OpenVINO Model Server target device, you need to:
- install the required drivers - refer to [OpenVINO installation guide](https://docs.openvino.ai/2022.1/openvino_docs_install_guides_installing_openvino_linux.html#step-5-optional-configure-inference-on-non-cpu-devices)
- start the docker container with the additional parameter of `--device /dev/dri` to pass the device context 
- set the parameter of `--target_device` to `GPU`.
- use the `openvino/model_server:latest-gpu` image, which contains GPU dependencies

A command example:

```bash

docker run --rm -it --device=/dev/dri -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU

```

Running inference on GPU requires the model server process security context account to have correct permissions. It must belong to the render group identified by the command:

```bash
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```

The default account in the docker image is preconfigured. If you change the security context, use the following command to start the model server container:

```bash

docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \

-v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \

--model_path /opt/model --model_name my_model --port 9001 --target_device GPU

```

> **NOTE**:
> The public docker image includes the OpenCL drivers for GPU in version 21.38.21026 (RedHat) and 21.48.21782 (Ubuntu).

Support for [Intel Arc](https://www.intel.com/content/www/us/en/architecture-and-technology/visual-technology/arc-discrete-graphics.html), which is in preview now, requires newer driver version `22.10.22597`. You can build OpenVINO Model server with ubuntu base image and that driver using the command below:
```bash
make docker_build INSTALL_DRIVER_VERSION=22.10.22597
```

## Using Multi-Device Plugin

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. 
It distributes Inference requests among multiple devices, balancing out the load. For more detailed information read OpenVINO’s [Multi-Device plugin documentation](https://docs.openvino.ai/2022.1/openvino_docs_OV_UG_Running_on_multiple_devices.html) documentation.

To use this feature in OpenVINO Model Server, you can choose one of two ways:

1. Use a .json configuration file to set the `--target_device` parameter with the pattern of: `MULTI:<DEVICE_1>,<DEVICE_2>`. 
The order of the devices will define their priority, in this case making `device_1` the primary selection. 

This example of a config.json file sets up the Multi-Device Plugin for a resnet model, using Intel Movidius Neural Compute Stick and CPU as devices:

```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "target_device": "MULTI:MYRIAD,CPU"}
   }]
}
```

To start OpenVINO Model Server, with the described config file placed as `./models/config.json`, set the `grpc_workers` parameter to match the `nireq` field in config.json 
and use the run command, like so:

```
docker run -d --net=host -u root --privileged --rm -v $(pwd)/models/:/opt/ml:ro -v /dev:/dev -p 9001:9001 \
openvino/model_server:latest --config_path /opt/ml/config.json --port 9001 
```

2. When using just a single model, you can start OpenVINO Model Server without the config.json file. To do so, use the run command together with additional parameters, like so: 

```
docker run -d --net=host -u root --privileged --name ie-serving --rm -v $(pwd)/models/:/opt/ml:ro -v \ 
/dev:/dev -p 9001:9001 openvino/model_server:latest model --model_path /opt/ml/resnet --model_name resnet --port 9001 --target_device 'MULTI:MYRIAD,CPU'
```
 
The deployed model will perform inference on both Intel Movidius Neural Compute Stick and CPU. 
The total throughput will be roughly equal to the sum of CPU and Intel Movidius Neural Compute Stick throughputs.
 

## Using Heterogeneous Plugin

The [HETERO plugin](https://docs.openvino.ai/2022.1/openvino_docs_OV_UG_Hetero_execution.html) makes it possible to distribute inference load of one model 
among several computing devices. That way different parts of the deep learning network can be executed by devices best suited to their type of calculations. 
OpenVINO automatically divides the network to optimize the process.

The Heterogenous plugin can be configured using the `--target_device` parameter with the pattern of: `HETERO:<DEVICE_1>,<DEVICE_2>`. 
The order of devices will define their priority, in this case making `device_1` the primary and `device_2` the fallback one.

Here is a config example using heterogeneous plugin with GPU as the primary device and CPU as a fallback.

```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "target_device": "HETERO:GPU,CPU"}
   }]
}
```

## Using AUTO Plugin

[Auto Device](https://docs.openvino.ai/2022.1/openvino_docs_IE_DG_supported_plugins_AUTO.html) (or AUTO in short) is a new special “virtual” or “proxy” device in the OpenVINO toolkit, it doesn’t bind to a specific type of HW device.
AUTO solves the complexity in application required to code a logic for the HW device selection (through HW devices) and then, on the deducing the best optimization settings on that device.
AUTO always chooses the best device, if compiling model fails on this device, AUTO will try to compile it on next best device until one of them succeeds.
Make sure you have passed the devices and access to the devices you want to use in for the docker image. For example with:
```--device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g)```

Below is an example of the command with AUTO Plugin as target device. It includes extra docker parameters to enable GPU (/dev/dri) , beside CPU.
@sphinxdirective

   .. code-block:: sh

        docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)\
            -u $(id -u):$(id -g) -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
            --model_path /opt/model --model_name my_model --port 9001 \
            --target_device AUTO

@endsphinxdirective

The `Auto Device` plugin can also use the [PERFORMANCE_HINT](performance_tuning.md) plugin config property that enables you to specify a performance mode for the plugin.

> **NOTE**: CPU_THROUGHPUT_STREAMS and PERFORMANCE_HINT should not be used together.

To enable Performance Hints for your application, use the following command:
@sphinxdirective

.. tab:: LATENCY  

   .. code-block:: sh

        docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
            -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
            --model_path /opt/model --model_name my_model --port 9001 \
            --plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
            --target_device AUTO

.. tab:: THROUGHTPUT

   .. code-block:: sh  
   
        docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
            -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
            --model_path /opt/model --model_name my_model --port 9001 \
            --plugin_config '{"PERFORMANCE_HINT": "THROUGHTPUT"}' \
            --target_device AUTO

@endsphinxdirective

> **NOTE**: currently, AUTO plugin cannot be used with `--shape auto` parameter while GPU device is enabled.
