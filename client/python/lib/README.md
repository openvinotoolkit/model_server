# OpenVINO&trade; Model Server Client Library

Model server client library is a set of objects and methods designed to simplify user interaction with the instance of the model server. The library contains functions that hide API specific aspects, so user doesn't have to know about creating protos, preparing requests, parsing responses etc. and can focus on the application itself, rather than dealing with all the aspects of the interaction with OVMS.

OVMS client library contains only the necessary dependencies, so the whole package is light. That makes it more friendly for deployments with restricted resources as well as for the use cases that require applications to scale well.

As OpenVINO Model Server API is compatibile with TensorFlow Serving, it's possible to use `ovmsclient` with TensorFlow Serving instances on: Predict, GetModelMetadata and GetModelStatus endpoints.

See [API documentation](docs/README.md) for details on what the library provides.


## Installation

The client library requires Python in version >= 3.6.

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

   `pip install dist/ovmsclient-0.1-py3-none-any.whl`

*Note*: For development purposes you may want to repeatedly reinstall the package.
For that consider using `pip install` with `--force-reinstall` and `--no-deps` options.

Apart from `make build`, there are also other targets available:
 - `make build-deps` - downloads and compiles TFS API protos
 - `make build-package` - builds only `ovmsclient` package (requires TFS API protos compiled)
 - `make test` - runs tests on `ovmsclient` package. By default the package located in `dist/` directory is used. To specify custom package path pass `PACKAGE_PATH` option like: 

   `make test PACKAGE_PATH=/opt/packages/ovmsclient-0.1-py3-none-any.whl`

 - `make clean` - removes all intermediate files generated while building the package


## Usage

**Create gRPC client instance:**
```python
import ovmsclient

config = {
   "address": "localhost", 
   "port": 9000
   }

client = ovmsclient.make_grpc_client(config="config")
```

**Create and send model status request:**
```python
status_request = ovmsclient.make_grpc_status_request(model_name="model")
status_response = client.get_model_status(status_request)
status_response.to_dict()

# Examplary status_response.to_dict() output:
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
metadata_request = ovmsclient.make_grpc_metadata_request(model_name="model")
metadata_response = client.get_model_metadata(metadata_request)
metadata_response.to_dict()

# Examplary metadata_response.to_dict() output. Values for model:
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
predict_request = ovmsclient.make_grpc_predict_request(
    { "map/TensorArrayStack/TensorArrayGatherV3": img },
    model_name="model")
predict_response = client.predict(predict_request)
predict_response.to_dict()

# Examplary predict_response.to_dict() output:
#
#{
#   "softmax_tensor": [[0.01, 0.03, 0.91, ... , 0.00021]]
#}
#
```

For more details on `ovmsclient` see [API reference](docs/README.md)