# CPU Extensions {#ovms_sample_cpu_extension}

Any CPU layer, unsupported by OpenVINO, can be implemented as a shared library. While loaded in OVMS as a cpu extension, it can help in executing the model. An example presented here is based on the code from in OpenVINO™ repository: [extension template](https://github.com/openvinotoolkit/openvino/tree/master/src/core/template_extension).

It includes a demonstrative implementation of the Relu layer which can be applied on many existing
public models. That implementation display in the model server logs information about the
custom extension execution.

## Creating cpu_extension library

Compile the library by running `make cpu_extension BASE_OS=ubuntu` in root directory of [Model Server repository](https://github.com/openvinotoolkit/model_server/tree/releases/2026/4). The implementation of this library slightly differs from the template in OpenVINO™ repository and can be found in [SampleCpuExtension directory](https://github.com/openvinotoolkit/model_server/tree/releases/2026/4/src/example/SampleCpuExtension).

Shared library will be generated in the `lib` folder. Such library can be used to run Model Server, using `--cpu_extension` argument.

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
# replace to 'redhat` if using UBI base image
export BASE_OS=ubuntu24
make cpu_extension BASE_OS=${BASE_OS}
```

## Preparing resnet50 model

In order to demonstrate the usage of cpu_extension library some small modifications in resnet model are needed.
In this sample we are going to change one of the ReLU layers type to CustomReLU.
By doing so this layer will take advantage of cpu_extension.

```bash
mkdir -p ${HOME}/models
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o ${HOME}/models/resnet50.xml
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o ${HOME}/models/resnet50.bin
sed -i '0,/ReLU/s//CustomReLU/' ${HOME}/models/resnet50.xml
```

## Deploying OVMS

```bash
$ docker run -it --rm -p 9000:9000 -v `pwd`/lib/${BASE_OS}:/extension:ro -v ${HOME}/models:/models openvino/model_server \
 --port 9000 --model_name resnet --model_path /models/resnet50.xml \
 --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" \
 --cpu_extension /extension/libcustom_relu_cpu_extension.so
```

> **NOTE**: Learn more about [OpenVINO extensibility](https://docs.openvino.ai/2026/documentation/openvino-extensibility.html)
