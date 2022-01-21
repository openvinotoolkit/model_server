## Sample CPU Extension (custom ReLU)

Any CPU layer, unsupported by OpenVINO, can be implemented as a shared library. While loaded in OVMS as a cpu extension, it can help in executing the model. An example presented here is based on the code from in OpenVINO™ repository: [extension template](https://github.com/openvinotoolkit/openvino/tree/master/docs/template_extension/new).

It includes a demonstrative implementation of the Relu layer which can be applied on many existing
public models. That implementation display in the model server logs information about the 
custom extension execution.

Compile the library by running `make cpu_extension BASE_OS=ubuntu` in root directory of this repository.

Shared library will be generated in the `lib` folder. Such library can be used to run Model Server, using `--cpu_extension` argument:

```bash
$ docker run -it --rm -p 9000:9000 -v <your_lib_dir>:/extension:ro openvino/model_server \
    --port 9000 --model_name resnet --model_path gs://ovms-public-eu/resnet50-binary  \
    --cpu_extension /extension/libcustom_relu_cpu_extension.so
```

> NOTE: Learn more about [OpenVINO extensibility](https://docs.openvino.ai/latest/openvino_docs_IE_DG_Extensibility_DG_Intro.html) 
