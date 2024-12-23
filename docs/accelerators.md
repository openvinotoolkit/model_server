# Using AI Accelerators {#ovms_docs_target_devices}

## Prepare test model

Download ResNet50 model

```bash
mkdir models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models openvino/ubuntu20_dev:latest omz_downloader --name resnet-50-tf --output_dir /models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models:rw openvino/ubuntu20_dev:latest omz_converter --name resnet-50-tf --download_dir /models --output_dir /models --precisions FP32
mv ${PWD}/models/public/resnet-50-tf/FP32 ${PWD}/models/public/resnet-50-tf/1
```

## Starting a Docker Container with Intel integrated GPU, Intel® Data Center GPU Flex Series and Intel® Arc™ GPU

The [GPU plugin](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) uses the Intel Compute Library for
Deep Neural Networks ([clDNN](https://01.org/cldnn)) to infer deep neural networks. For inference execution, it employs Intel® Processor Graphics including
Intel® HD Graphics, Intel® Iris® Graphics, Intel® Iris® Xe Graphics, and Intel® Iris® Xe MAX graphics.


Before using GPU as OpenVINO Model Server target device, you need to:
- install the required drivers - refer to [OpenVINO installation guide](https://docs.openvino.ai/2024/get-started/configurations.html)
- start the docker container with the additional parameter of `--device /dev/dri` to pass the device context
- set the parameter of `--target_device` to `GPU`.
- use the `openvino/model_server:latest-gpu` image, which contains GPU dependencies

Running inference on GPU requires the model server process security context account to have correct permissions. It must belong to the render group identified by the command:

```bash
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```

The default account in the docker image is preconfigured. If you change the security context, use the following command to start the model server container:

```bash
docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9001 --target_device GPU
```

GPU device can be used also on Windows hosts with Windows Subsystem for Linux 2 (WSL2). In such scenario, there are needed extra docker parameters. See the command below.
Use device `/dev/dxg` instead of `/dev/dri` and mount the volume `/usr/lib/wsl`:

```bash
docker run --rm -it  --device=/dev/dxg --volume /usr/lib/wsl:/usr/lib/wsl -u $(id -u):$(id -g) \
-v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9001 --target_device GPU
```



## Using NPU device Plugin

OpenVINO Model Server can support using [NPU device](https://docs.openvino.ai/canonical/openvino_docs_install_guides_configurations_for_intel_npu.html)

Docker image with required dependencies can be build using this procedure:
The docker image of OpenVINO Model Server including support for NVIDIA can be built from sources

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make release_image NPU=1
cd ..
```

Example command to run container with NPU:
```bash
docker run --device /dev/accel -p 9000:9000 --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/models/public/resnet-50-tf:/opt/model openvino/model_server:latest --model_path /opt/model --model_name resnet --port 9000 --target_device NPU
```
Check more info about the [NPU driver for Linux](https://github.com/intel/linux-npu-driver).



## Using Heterogeneous Plugin

The [HETERO plugin](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html) makes it possible to distribute inference load of one model
among several computing devices. That way different parts of the deep learning network can be executed by devices best suited to their type of calculations.
OpenVINO automatically divides the network to optimize the process.

The Heterogeneous plugin can be configured using the `--target_device` parameter with the pattern of: `HETERO:<DEVICE_1>,<DEVICE_2>`.
The order of devices will define their priority, in this case making `device_1` the primary and `device_2` the fallback one.

Here is a config example using heterogeneous plugin with GPU as the primary device and CPU as a fallback.

```bash
echo '{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/model",
      "batch_size": "1",
      "target_device": "HETERO:GPU,CPU"}
   }]
}' >> models/public/resnet-50-tf/config.json
```

## Using AUTO Plugin

[Auto Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html) (or AUTO in short) is a new special “virtual” or “proxy” device in the OpenVINO toolkit, it doesn’t bind to a specific type of HW device.
AUTO solves the complexity in application required to code a logic for the HW device selection (through HW devices) and then, on the deducing the best optimization settings on that device.
AUTO always chooses the best device, if compiling model fails on this device, AUTO will try to compile it on next best device until one of them succeeds.
Make sure you have passed the devices and access to the devices you want to use in for the docker image. For example with:
`--device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g)`

Below is an example of the command with AUTO Plugin as target device. It includes extra docker parameters to enable GPU (/dev/dri) , beside CPU.

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
-u $(id -u):$(id -g) -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9001 \
--target_device AUTO
```

The `Auto Device` plugin can also use the [PERFORMANCE_HINT](performance_tuning.md) plugin config property that enables you to specify a performance mode for the plugin.
While LATENCY and THROUGHPUT hint can select one target device with your preferred performance option, the CUMULATIVE_THROUGHPUT option enables running inference on multiple devices for higher throughput. 
With CUMULATIVE_THROUGHPUT, AUTO loads the network model to all available devices (specified by AUTO) in the candidate list, and then runs inference on them based on the default or specified priority.

> **NOTE**: NUM_STREAMS and PERFORMANCE_HINT should not be used together.

To enable Performance Hints for your application, use the following command:

LATENCY

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9001 \
--plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
--target_device AUTO
```

THROUGHPUT

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9001 \
--plugin_config '{"PERFORMANCE_HINT": "THROUGHPUT"}' \
--target_device AUTO
```

CUMULATIVE_THROUGHPUT

```bash
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9001 \
--plugin_config '{"PERFORMANCE_HINT": "CUMULATIVE_THROUGHPUT"}' \
--target_device AUTO:GPU,CPU
```


## Using Automatic Batching Plugin

[Auto Batching](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching.html) (or BATCH in short) is a new special “virtual” device 
which explicitly defines the auto batching. 

It performs automatic batching on-the-fly to improve device utilization by grouping inference requests together, without programming effort from the user. 
With Automatic Batching, gathering the input and scattering the output from the individual inference requests required for the batch happen transparently, without affecting the application code.

> **NOTE**: Autobatching can be applied only for static models

```bash
docker run -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name resnet --port 9001 \
--plugin_config '{"AUTO_BATCH_TIMEOUT": 200}' \
--target_device BATCH:CPU(16)
```