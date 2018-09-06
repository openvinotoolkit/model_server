# sagemaker-tensorflow-containers with OpenVINO Model Server for inference execution

## Overview
Sagemaker-tensorflow-containers is a component which exposes REST API interface for inference execution including
pre and post processing. 
It was designed for deploying TensorFlow framework in Amazon but it could be used also as a standalone local docker container. 

In the backend, sagemaker-tensorflow-containers is using TensorFlow Serving component for inference execution over gRPC interface.
While this works correctly it is not performing in the optimal manner on the CPU. Based on our tests with various AI graph 
topologies, employing OpenVINO and its Inference Engine can speed up the inference processing over 3 times on the same HW!

The solution presented in this example is swapping TensorFlow Serving with OpenVINO Model Server inside the Sagemaker container.

Because the gRPC API is compatible between those 2 component the replacement is pretty transparent. There is just needed
a patch in the code altering the service startup command and instead of TensorFlow saved models format there should 
be used intermediate representation format, which can be converted by
[OpenVINO model optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer). 


## Building the docker image

The first step in the building process should applying the git patch 
[0001-patch-sagemaker-for-OpenVINO-model-server.patch](0001-patch-sagemaker-for-OpenVINO-model-server.patch) 
on the original repository https://github.com/aws/sagemaker-tensorflow-container
 (tested with a commit 1456108c6c025a6b81cf0de462522c8517b3005e).
It is updating the command for staring the model server from TensorFlow Serving to OpenVINO Model Server.

```bash
git apply 0001-patch-sagemaker-for-OpenVINO-model-server.patch
```

The next step should be creating SageMaker TensorFlow Container Python package according to the official documentation:

```bash
# Create the SageMaker TensorFlow Container Python package.
cd sagemaker-tensorflow-containers
python setup.py sdist

#. Copy your Python package to “example_sagemaker” folder with the Dockerfile.
cp dist/sagemaker_tensorflow_container-*.tar.gz [path]/example_sagemeker/
```

It is also required to download OpenVINO installer package from https://software.intel.com/en-us/openvino-toolkit/choose-download 
The Linux or Linux FPGA versions can be used.
It is recommended to download online installer as it results in a smaller docker image size. Place the installer 
in the same folder like the Dockerfile and Sagemaker python package.

Eventually build the docker image with a command:
```bash
docker build --build-arg HTTP_PROXY="$http_proxy" --build-arg HTTPS_PROXY="$http_proxy" -f Dockerfile -t sagemaker-ie_serving .
```

With this step docker image is build and ready to be used.


## Starting the sagemaker container

In the example below sagemaker docker container starts serving models from folder `/tmp/test_models/saved_models` on port 
8080:
```bash

docker run --rm -d -v /tmp/test_models/saved_models/:/opt/ml/:ro -p 8080:8080 --env-file=env.txt -t sagemaker-ie_serving serve
```

With that the predictions calls can be executed in a way similar to:

```python
serialized_output = requests.post("http://localhost:8080/invocations",
                                          MessageToJson(tensor_proto),
                                          headers={'Content-type': 'application/json'}).content
```
or
```python
serialized_output = requests.post("http://localhost:8080/invocations",
                                      data=request.SerializeToString(),
                                      headers={
                                          'Content-type': 'application/octet-stream',
                                          'Accept': 'application/octet-stream'
                                      }).content
```