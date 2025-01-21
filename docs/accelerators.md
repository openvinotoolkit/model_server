# Using AI Accelerators {#ovms_docs_target_devices}

## Prerequisites

Docker engine installed (on Linux and WSL), or ovms binary package installed as described in the [guide](./deploying_server_baremetal.md) (on Linux or Windows). 

Supported HW is documented in [OpenVINO system requirements](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino/system-requirements.html)

Before staring the model server as a binary package, make sure there are installed GPU or/and NPU required drivers like described in [https://docs.openvino.ai/2024/get-started/configurations.html](https://docs.openvino.ai/2024/get-started/configurations.html)

Additional considerations when deploying with docker container:
- make sure to use the image version including runtime drivers. The public image has a suffix -gpu like `openvino/model_server:latest-gpu`.
- additional parameters needs to be passed to the docker run command depending on the accelerator.
- kernel modules needs to be present on the host with support for the accelerators

## Prepare test model

The model format for the deployment is identical regardless of the employed accelerators. It is compiled in runtime based on the parameter `target_device`.
It can be set in the model server CLI, in the configuration files `config.json` or `subconfig.json`. When using pipelines based on MediaPipe graphs, there might be also additional parameters defined in `graph.pbtxt` files.

For example, download ResNet50 model to follow the guide below:
```console
curl -L -o model/1/model.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/resnet-50/tensorFlow2/classification/1/download --create-dirs
tar -xzf model/1/model.tar.gz -C model/1/
rm model/1/model.tar.gz
```

## Starting Model Server with Intel GPU

The [GPU plugin](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) uses the [oneDNN](https://github.com/oneapi-src/oneDNN) and [OpenCL](https://github.com/KhronosGroup/OpenCL-SDK) to infer deep neural networks. For inference execution, it employs Intel® Processor Graphics including
Intel® Arc™ GPU Series, Intel® UHD Graphics, Intel® HD Graphics, Intel® Iris® Graphics, Intel® Iris® Xe Graphics, and Intel® Iris® Xe MAX graphics and Intel® Data Center GPU.

### Container

Running inference on GPU requires the model server process security context account to have correct permissions. It must belong to the render group identified by the command:

```bash
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```

Use the following command to start the model server container:

```bash
docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9000 --target_device GPU
```

GPU device can be used also on Windows hosts with Windows Subsystem for Linux 2 (WSL2). In such scenario, there are needed extra docker parameters. See the command below.
Use device `/dev/dxg` instead of `/dev/dri` and mount the volume `/usr/lib/wsl`:

```bash
docker run --rm -it  --device=/dev/dxg --volume /usr/lib/wsl:/usr/lib/wsl -u $(id -u):$(id -g) \
-v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9000 --target_device GPU
```

### Binary 

Starting the server with GPU acceleration requires installation of runtime drivers and ocl-icd-libopencl1 package like described on [configuration guide](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html)

Start the model server with GPU accelerations using a command:
```console
ovms --model_path model --model_name resnet --port 9000 --target_device GPU
```


## Using NPU device Plugin

OpenVINO Model Server supports using [NPU device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

### Container
Example command to run container with NPU:
```bash
docker run --device /dev/accel -p 9000:9000 --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/model:/opt/model openvino/model_server:latest-gpu --model_path /opt/model --model_name resnet --port 9000 --target_device NPU
```

### Binary package
Start the model server with NPU accelerations using a command:
```console
ovms --model_path model --model_name resnet --port 9000 --target_device NPU
```

Check more info about the [NPU driver configuration](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-npu.html).



## Using Heterogeneous Plugin

The [HETERO plugin](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html) makes it possible to distribute inference load of one model
among several computing devices. That way different parts of the deep learning network can be executed by devices best suited to their type of calculations.
OpenVINO automatically divides the network to optimize the process.

The Heterogeneous plugin can be configured using the `--target_device` parameter with the pattern of: `HETERO:<DEVICE_1>,<DEVICE_2>`.
The order of devices will define their priority, in this case making `device_1` the primary and `device_2` the fallback one.

Here is a config example using heterogeneous plugin with GPU as the primary device and CPU as a fallback.

### Container

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
-u $(id -u):$(id -g) -v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9000 \
--target_device "HETERO:GPU,CPU"
```

### Binary package

```console
ovms --model_path model --model_name resnet --port 9000 --target_device "HETERO:GPU,CPU"
```


## Using AUTO Plugin

[Auto Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html) (or AUTO in short) is a new special “virtual” or “proxy” device in the OpenVINO toolkit, it doesn’t bind to a specific type of HW device.
AUTO solves the complexity in application required to code a logic for the HW device selection (through HW devices) and then, on the deducing the best optimization settings on that device.
AUTO always chooses the best device, if compiling model fails on this device, AUTO will try to compile it on next best device until one of them succeeds.

The `Auto Device` plugin can also use the [PERFORMANCE_HINT](performance_tuning.md) plugin config property that enables you to specify a performance mode for the plugin.
While LATENCY and THROUGHPUT hint can select one target device with your preferred performance option, the CUMULATIVE_THROUGHPUT option enables running inference on multiple devices for higher throughput. 
With CUMULATIVE_THROUGHPUT hint, AUTO plugin loads the network model to all available devices (specified by the plugin) in the candidate list, and then runs inference on them based on the default or specified priority.

> **NOTE**: NUM_STREAMS and PERFORMANCE_HINT should not be used together.

### Container

Make sure you have passed the devices and access to the devices you want to use in for the docker image. For example with:
`--device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g)`

Below is an example of the command with AUTO Plugin as target device. It includes extra docker parameters to enable GPU (/dev/dri) , beside CPU.

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
-u $(id -u):$(id -g) -v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9000 \
--target_device AUTO
```

To enable Performance Hints for your application, use the following command:

LATENCY

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9000 \
--plugin_config "{\"PERFORMANCE_HINT\": \"LATENCY\"}" \
--target_device AUTO
```

THROUGHPUT

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9000 \
--plugin_config "{\"PERFORMANCE_HINT\": \"THROUGHPUT\"}" \
--target_device AUTO
```

CUMULATIVE_THROUGHPUT

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9000 \
--plugin_config '{"PERFORMANCE_HINT": "CUMULATIVE_THROUGHPUT"}' \
--target_device AUTO:GPU,CPU
```

### Binary package

Below is the equivalent of the deployment command with a binary package at below:

AUTO
```console
ovms --model_path model --model_name resnet --port 9000 --target_device AUTO:GPU,CPU
```

THROUGHPUT
```console
ovms --model_path model --model_name resnet --port 9000 --plugin_config "{\"PERFORMANCE_HINT\": \"THROUGHPUT\"}" --target_device AUTO:GPU,CPU
```

LATENCY
```console
ovms --model_path model --model_name resnet --port 9000 --plugin_config "{\"PERFORMANCE_HINT\": \"LATENCY\"}" --target_device AUTO:GPU,CPU
```

CUMULATIVE_THROUGHPUT
```console
ovms --model_path model --model_name resnet --port 9000 --plugin_config "{\"PERFORMANCE_HINT\": \"CUMULATIVE_THROUGHPUT\"}" --target_device AUTO:GPU,CPU
```

## Using Automatic Batching Plugin

[Auto Batching](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching.html) (or BATCH in short) is a new special “virtual” device 
which explicitly defines the auto batching. 

It performs automatic batching on-the-fly to improve device utilization by grouping inference requests together, without programming effort from the user. 
With Automatic Batching, gathering the input and scattering the output from the individual inference requests required for the batch happen transparently, without affecting the application code.

> **NOTE**: Autobatching can be applied only for static models
> **NOTE**: Autobatching is enabled by default, then the `target_device` is set to `GPU` with `--plugin_config '{"PERFORMANCE_HINT": "THROUGHPUT"}'`. 

### Containers
```bash
docker run -v ${PWD}/model:/opt/model -p 9000:9000 openvino/model_server:latest \
--model_path /opt/model --model_name resnet --port 9000 \
--plugin_config '{"AUTO_BATCH_TIMEOUT": 200}' \
--target_device "BATCH:CPU(16)"
```
In the example above, there will be 200ms timeout to wait for filling the batch size up to 16.

### Binary package

The same deployment with a binary package can be completed with a command:
```console
ovms --model_path model --model_name resnet --port 9000 --plugin_config "{\"AUTO_BATCH_TIMEOUT\": 200}" --target_device "BATCH:CPU(16)"
```
```
