# OpenVINO&trade; Model Server Client Library

Model server client library is a set of objects and methods designed to simplify user interaction with the instance of the model server. The library contains functions that hide API specific aspects, so user doesn't have to know about creating protos, preparing requests, parsing responses etc. and can focus on the application itself, rather than dealing with all the aspects of the interaction with OVMS.

OVMS client library contains only the necessary dependencies, so the whole package is light. That makes it more friendly for deployments with restricted resources as well as for the use cases that require applications to scale well.

As OpenVINO Model Server API is compatible with TensorFlow Serving, it's possible to use `ovmsclient` with TensorFlow Serving instances on: Predict, GetModelMetadata and GetModelStatus endpoints.

See [API documentation](docs/README.md) for details on what the library provides.


## Installation

**Note:** The client library requires Python in version >= 3.6.

Install the `ovmsclient` package with:

`pip install ovmsclient`

You can also build the wheel from sources:

### Linux

Prerequisites:
 - Python 3.6 +
 - Python package [setuptools](https://pypi.org/project/setuptools/)
 - Protoc 3.6.1 + [protobuf-compiler](https://grpc.io/docs/protoc-installation/)
 We recommend to install [pre-compiled binaries](https://grpc.io/docs/protoc-installation/#install-pre-compiled-binaries-any-os) to get newest protoc version

**To build the package run:**

   `make build`

This command will create pip wheel placed in `dist` directory.

*Note*: For development purposes, you may want to repeatedly rebuild the package.
Assuming you have TFS API built, you can use `make build-package` target to build only the `ovmsclient` package and ommit downloading and building the TFS API.

**To install the package run:**

   `pip install dist/ovmsclient-0.2-py3-none-any.whl`

*Note*: For development purposes you may want to repeatedly reinstall the package.
For that consider using `pip install` with `--force-reinstall` and `--no-deps` options.

Apart from `make build`, there are also other targets available:
 - `make build-deps` - downloads and compiles TFS API protos
 - `make build-package` - builds only `ovmsclient` package (requires TFS API protos compiled)
 - `make test` - runs tests on `ovmsclient` package. By default the package located in `dist/` directory is used. To specify custom package path pass `PACKAGE_PATH` option like: 

   `make test PACKAGE_PATH=/opt/packages/ovmsclient-0.2-py3-none-any.whl`

 - `make clean` - removes all intermediate files generated while building the package


## Use in Docker container

There are also Dockerfiles available that prepare Docker image with `ovmsclient` installed and ready to use.
Simply run `docker build` with the Dockerfile of your choice to get the minimal image:
- [Ubuntu 20.04 based image](../Dockerfile.ubuntu)
- [RHEL 8.4 based image](../Dockerfile.redhat)

## Usage

**Create gRPC client instance:**
```python
import ovmsclient

client = ovmsclient.make_grpc_client("localhost:9000")
```

**Create and send model status request:**
```python
model_status = client.get_model_status(model_name="model")

# Examplary status_response:
#
# {
#    "1": {
#        "state": "AVAILABLE", 
#        "error_code": 0, 
#        "error_message": ""
#    }             
# } 
#
```

**Create and send model metadata request:**
```python
model_metadata = client.get_model_metadata(model_name="model")

# Exemplary metadata_response. Values for model:
# https://docs.openvinotoolkit.org/latest/omz_models_model_resnet_50_tf.html
#
#{
#   "1": {
#       "inputs": {
#           "map/TensorArrayStack/TensorArrayGatherV3": {
#               "shape": [1, 224, 224, 3],
#               "dtype": DT_FLOAT32  
#           }
#       },
#       "outputs" {
#           "softmax_tensor": {
#               "shape": [1, 1001],
#               "dtype": DT_FLOAT32  
#           }
#       }
#   }
#}
#
```

**Create and send predict request with binary input data:**
```python
# Assuming requesting model with inputs and outputs as in:
# https://docs.openvinotoolkit.org/latest/omz_models_model_resnet_50_tf.html

with open(<path_to_img>, 'rb') as f:
    img = f.read()
inputs = {"map/TensorArrayStack/TensorArrayGatherV3": img}
results = client.predict(inputs=inputs, model_name="model")

# Examplary results:
#
# [[0.01, 0.03, 0.91, ... , 0.00021]]
#
```

For more details on `ovmsclient` see [API reference](docs/README.md)