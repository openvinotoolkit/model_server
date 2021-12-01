# Using AI accelerators {#ovms_docs_target_devices}

### Running OpenVINO&trade; Model Server with AI Accelerators NCS, HDDL and GPU <a name="ai"></a>

<details><summary>Using an Intel Movidius Neural Compute Stick</summary>

#### Prepare to use an Intel Movidius Neural Compute Stick

[Intel® Movidius Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick) can be employed by OVMS via a [MYRIAD
plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_MYRIAD.html). 

The Intel Movidius Neural Compute Stick must be visible and accessible on host machine. 

NCS devices should be reported by `lsusb` command, which should print out `ID 03e7:2485`.<br>


#### Start the server with an Intel Movidius Neural Compute Stick

To start server with Neural Compute Stick using one of the options below:
1) More secure without docker privileged mode and mounting only the usb devices:
```
docker run --rm -it -u 0 --device-cgroup-rule='c 189:* rmw' -v /opt/model:/opt/model -v /dev/bus/usb:/dev/bus/usb -p 9001:9001 openvino/model_server \
--model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```
2) less secure in docker privileged mode wth mounted all devices:
```
docker run --rm -it --net=host -u root --privileged -v /opt/model:/opt/model -v /dev:/dev -p 9001:9001 openvino/model_server \
--model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```

</details>

<details><summary>Starting docker container with HDDL</summary>

In order to run container that is using HDDL accelerator, _hddldaemon_ must
 run on host machine. It's required to set up environment 
 (the OpenVINO package must be pre-installed) and start _hddldaemon_ on the
 host before starting a container. Refer to the steps from [OpenVINO documentation](https://docs.openvinotoolkit.org/2021.4/_docs_install_guides_installing_openvino_docker_linux.html#build_docker_image_for_intel_vision_accelerator_design_with_intel_movidius_vpus).

To start server with HDDL you can use command similar to:

```
docker run --rm -it --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```

`--device=/dev/ion:/dev/ion` mounts the accelerator device.

`-v /var/tmp:/var/tmp` enables communication with _hddldaemon_ running on the
 host machine

Check out our recommendations for [throughput optimization on HDDL](performance_tuning.md#hddl-accelerators)

*Note:* OpenVINO Model Server process in the container communicates with hddldaemon via unix sockets in /var/tmp folder.
It requires RW permissions in the docker container security context. It is recommended to start docker container in the
same context like the account starting hddldaemon. For example if you start the hddldaemon as root, add `--user root` to 
the `docker run` command.

</details>

<details><summary>Starting docker container with GPU</summary>

The [GPU plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_GPU.html) uses the Intel® Compute Library for Deep Neural Networks ([clDNN](https://01.org/cldnn)) to infer deep neural networks. 
It employs for inference execution Intel Processor Graphics including Intel® HD Graphics and Intel Iris Graphics.

Before using GPU as OVMS target device, you need to install the required drivers. Refer to [OpenVINO installation steps](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_linux.html).
Next, start the docker container with additional parameter --device /dev/dri to pass the device context and set OVMS parameter --target_device GPU. 
The command example is listed below:

```
docker run --rm -it --device=/dev/dri -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

Running the inference operation on GPU requires the ovms process security context account to have correct permissions.
It has to belong to the render group identified by the command:
```
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```
The default account in the docker image is already preconfigured. In case you change the security context, use the following command
to start the ovms container:
```
docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) -u $(id -u):$(id -g) \
-v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

*Note:* The public docker image includes the OpenCL drivers for GPU in version 20.35.17767. Older version could be used 
via building the image with a parameter INSTALL_DRIVER_VERSION:
`make docker_build INSTALL_DRIVER_VERSION=19.41.14441`. The older GPU driver will not support the latest Intel GPU platforms like TigerLake or newer.
</details>

<details><summary>Using Multi-Device Plugin</summary>

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. 
With Multi-Device Plugin enabled, inference requests will be load balanced between multiple devices. 
For more detailed information read [OpenVino's Multi-Device plugin documentation](https://docs.openvinotoolkit.org/2021.4/_docs_IE_DG_supported_plugins_MULTI.html).

In order to use this feature in OpenVino Model Server, following steps are required:

Set target_device for the model in configuration json file to MULTI:<DEVICE_1>,<DEVICE_2> (e.g. MULTI:MYRIAD,CPU, order of the devices defines their priority, so MYRIAD devices will be used first in this example)

Below is exemplary config.json setting up Multi-Device Plugin for resnet model, using Intel® Movidius Neural Compute Stick and CPU devices:

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
Starting OpenVINO Model Server with config.json (placed in ./models/config.json path) defined as above, and with grpc_workers parameter set to match nireq field in config.json:
```
docker run -d --net=host -u root --privileged --rm -v $(pwd)/models/:/opt/ml:ro -v /dev:/dev -p 9001:9001 \
openvino/model_server:latest --config_path /opt/ml/config.json --port 9001 
```
Or alternatively, when you are using just a single model, start OpenVINO Model Server using this command (config.json is not needed in this case):
```
docker run -d --net=host -u root --privileged --name ie-serving --rm -v $(pwd)/models/:/opt/ml:ro -v \
 /dev:/dev -p 9001:9001 openvino/model_server:latest model --model_path /opt/ml/resnet --model_name resnet --port 9001 --target_device 'MULTI:MYRIAD,CPU'
 ```
After these steps, deployed model will perform inference on both Intel Movidius Neural Compute Stick and CPU.
Total throughput will be roughly equal to sum of CPU and Intel® Movidius Neural Compute Stick throughput.

</details>

<details><summary>Using Heterogeneous Plugin</summary>

[HETERO plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_HETERO.html) makes it possible to distribute a single inference processing and model between several AI accelerators.
That way different parts of the DL network can split and executed on optimized devices.
OpenVINO automatically divides the network to optimize the execution.

Similarly to the MULTI plugin, Heterogenous plugin can be configured by using `--target_device` parameter using the pattern: `HETERO:<DEVICE_1>,<DEVICE_2>`.
The order of devices defines their priority. The first one is the primary device while the second is the fallback.<br>
Below is a config example using heterogeneous plugin with GPU as a primary device and CPU as a fallback.

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
</details>

