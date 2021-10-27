## Sample CPU Extension (custom ReLU)

Any CPU layer, unsupported by OpenVINO, can be implemented as a shared library. While loaded in OVMS as a cpu extension, it can help in executing the model. Example in OpenVINO™ repository: [hello_reshape_ssd](https://github.com/openvinotoolkit/openvino/tree/master/inference-engine/samples/hello_reshape_ssd).

With some modifications the library can be used in OpenVINO™ Model Server as well. This directory contains the same example but adjusted to generate a shared cpu extension library.

Compile the library by running `make cpu_extension BASE_OS=ubuntu` in root directory of this repository.

Shared library will be generated for targeted OS. Such library can be used to run Model Server, using `--cpu_extension` argument:

```bash
$ docker run -it --rm -p 9000:9000 -v <your_lib_dir>:/extension:ro -v <your_model_dir>:/models:ro openvino/model_server \
    --port 9000 --model_name resnet --model_path /models/resnet50-binary \
    --cpu_extension /extension/libcustom_relu_cpu_extension.so
```

> NOTE: To see the library getting executed you need a model network with custom layer included. Use OpenVINO™ documentation to generate such model in IR format. [Customize Model Optimizer for unsupported layers](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html).
