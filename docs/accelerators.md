# Using AI accelerators {#ovms_docs_target_devices}

## Start the Server with a Neural Compute Stick

Using the [Intel Movidius Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick) with the server is possible with the [MYRIAD
plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_MYRIAD.html). 

The Intel Movidius Neural Compute Stick must be visible and accessible on the host machine. 

NCS devices should be reported by `lsusb` command, which should print out `ID 03e7:2485`.<br>

To start the server with Neural Compute Stick, use one of the options below:

1) More secure option: without docker privileged mode and mounting only the USB devices:
   ```
   docker run --rm -it -u 0 --device-cgroup-rule='c 189:* rmw' -v /opt/model:/opt/model -v /dev/bus/usb:/dev/bus/usb -p 9001:9001 openvino/model_server \
   --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
   ```

2) Less secure option: in docker privileged mode with mounting all devices:
   ```
   docker run --rm -it --net=host -u root --privileged -v /opt/model:/opt/model -v /dev:/dev -p 9001:9001 openvino/model_server \
   --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
   ```

## Starting a Docker Container with HDDL

A container using HDDL accelerator, _hddldaemon_ must
 run on the host machine. Set up the environment 
 (pre-install the OpenVINO package) and start _hddldaemon_ on the
 host before starting a container. Refer to the steps from [OpenVINO documentation](https://docs.openvinotoolkit.org/2021.4/_docs_install_guides_installing_openvino_docker_linux.html#build_docker_image_for_intel_vision_accelerator_design_with_intel_movidius_vpus).

To start the server with HDDL device(s) you can use a command similar to the following:

```
docker run --rm -it --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```

`--device=/dev/ion:/dev/ion` mounts the accelerator device.

`-v /var/tmp:/var/tmp` enables communication with _hddldaemon_ running on the
 host machine

Check out our recommendations for [throughput optimization on HDDL](performance_tuning.md#hddl-accelerators)

>Note: OpenVINO Model Server process in the container communicates with hddldaemon via unix sockets in /var/tmp folder.
It requires RW permissions in the docker container security context. It is recommended to start docker container in the
same context like the account starting hddldaemon. For example if you start the hddldaemon as root, add `--user root` to 
the `docker run` command.

## Starting a Docker Container with Intel GPU

The [GPU plugin](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_GPU.html) uses the Intel Compute Library for Deep Neural Networks ([clDNN](https://01.org/cldnn)) to infer deep neural networks. 
The plugin supports inference execution with Intel® Processor Graphics including Intel® HD Graphics, Intel® Iris® Graphics, Intel® Iris® Xe Graphics, and Intel® Iris® Xe MAX graphics.

Before using Intel GPU as a target device, you need to install the required drivers. Refer to [OpenVINO installation guide](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-gpu) for more details.
Next, start the docker container with an additional parameter `--device /dev/dri` (to pass the device context) and set parameter `--target_device GPU`. You will also need to use the `openvino/model_server:latest-gpu` image, which contains GPU dependencies. 
See an example command below:

```
docker run --rm -it --device=/dev/dri -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

Running inference on GPU device(s) requires the server process security context account to have the correct permissions.
It must belong to the render group identified by the following command:
```
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```
The default account in the Docker image `openvino/model_server:latest-gpu` is preconfigured. In case you change the security context, use the following command
to start the Model Server container:
```
docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) -u $(id -u):$(id -g) \
-v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

*Note:* The Docker image (`openvino/model_server:latest-gpu`) includes OpenCL for GPU driver version 20.35.17767. Older versions can be used 
by building the image from the source with a parameter `INSTALL_DRIVER_VERSION:`
`make docker_build INSTALL_DRIVER_VERSION=19.41.14441`. The older GPU drivers are not supported with the latest Intel® Iris® Xe Graphics, and Intel® Iris® Xe MAX graphics (starting with codename Tiger Lake).

## Using Multi-Device Plugin

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. 
With Multi-Device Plugin enabled, inference requests will be load balanced between multiple devices. 
For more details, see the [Multi-Device Plugin](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html) documentation.

To use this feature in OpenVINO Model Server, the following steps are required:

Set `target_device` for specific model(s) in the JSON configuration file to `MULTI:<DEVICE_1>,<DEVICE_2>` (e.g. `MULTI:MYRIAD,CPU`, where the order of the devices defines priority, so MYRIAD devices will be used first in this example)

An example `config.json` is included below. It shows how to set up Multi-Device Plugin for a sample `resnet` model, using Intel® Movidius Neural Compute Stick and CPU devices:

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
To start OpenVINO Model Server with a `config.json` file (placed in `./models/config.json` path) as defined above, with `grpc_workers` parameter set to match `nireq` field in `config.json`:
```
docker run -d --net=host -u root --privileged --rm -v $(pwd)/models/:/opt/ml:ro -v /dev:/dev -p 9001:9001 \
openvino/model_server:latest --config_path /opt/ml/config.json --port 9001 
```
Alternatively, when deploying a single model, start the server using the following command (`config.json` is not required for single model deployments):
```
docker run -d --net=host -u root --privileged --name ie-serving --rm -v $(pwd)/models/:/opt/ml:ro -v \
 /dev:/dev -p 9001:9001 openvino/model_server:latest model --model_path /opt/ml/resnet --model_name resnet --port 9001 --target_device 'MULTI:MYRIAD,CPU'
 ```
After these steps, deployed model will perform inference on both Intel Movidius Neural Compute Stick and CPU.
Total throughput will be roughly equal to sum of CPU and Intel Movidius Neural Compute Stick throughput.

## Using Heterogeneous Plugin

[HETERO plugin](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_HETERO.html) enables computing the inference of one network on several devices.
This enables splitting operations in a model and executed on different devices.
OpenVINO automatically divides the network to optimize the execution.

Similar to the MULTI plugin, HETERO plugin can be configured using `--target_device` parameter with the following pattern: `HETERO:<DEVICE_1>,<DEVICE_2>`.
The order of devices defines their priority. The first is the primary device and the second is the fallback.
The example configuration below uses the HETERO plugin with Intel GPU as the primary device and CPU as a fallback.

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
