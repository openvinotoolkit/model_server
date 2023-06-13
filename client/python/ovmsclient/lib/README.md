# ovmsclient python library {#ovms_client_python_lib_readme}

The Model Server client library is a set of objects and methods designed to simplify user interaction with instances of OpenVINO Model Server. The library contains functions that conceal API-specific details. Users do not need to know about creating protos, preparing requests, parsing responses, and other details, so they can focus on deploying the application.

OVMS client library contains only the necessary dependencies, so the whole package is light. That makes it more friendly for deployments with restricted resources as well as for the use cases that require applications to scale well.

As OpenVINO Model Server API is compatible with TensorFlow Serving, it's possible to use `ovmsclient` with TensorFlow Serving instances on: Predict, GetModelMetadata and GetModelStatus endpoints.

See [API documentation](https://github.com/openvinotoolkit/model_server/blob/releases/2022/3/client/python/ovmsclient/lib/docs/README.md) for details on what the library provides.

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/ovmsclient/lib
```

## Installation

**Note:** The client library requires Python in version >= 3.6.

```bash
pip3 install ovmsclient
```

## Build the wheel

### Linux

Prerequisites:
 - Python 3.6 +
 - Python package [setuptools](https://pypi.org/project/setuptools/)
 - Protoc 3.19.2 + [protobuf-compiler](https://grpc.io/docs/protoc-installation/)
 We recommend to install [pre-compiled binaries](https://grpc.io/docs/protoc-installation/#install-pre-compiled-binaries-any-os) to get newest protoc version

**To build the package run:**

```bash
make build
```

This command will create pip wheel placed in `dist` directory.

*Note*: For development purposes, you may want to repeatedly rebuild the package.
Assuming you have TFS API built, you can use `make build-package` target to build only the `ovmsclient` package and omit downloading and building the TFS API.

**To install the package run:**
```bash
pip3 install --force-reinstall --no-deps dist/ovmsclient-2023.0-py3-none-any.whl
```

*Note*: For development purposes you may want to repeatedly reinstall the package.
For that consider using `pip3 install` with `--force-reinstall` and `--no-deps` options.

Apart from `make build`, there are also other targets available:
 - `make build-deps` - downloads and compiles TFS API protos
```bash
make build-deps
```
 - `make build-package` - builds only `ovmsclient` package (requires TFS API protos compiled)
 ```bash
make build-package
```
 - `make test` - runs tests on `ovmsclient` package. By default the package located in `dist/` directory is used. To specify custom package path pass `PACKAGE_PATH` option like: 

   `make test PACKAGE_PATH=/opt/packages/ovmsclient-2023.0-py3-none-any.whl`
```bash
make test
```
 - `make clean` - removes all intermediate files generated while building the package
```bash
make clean
```

## Usage

**Create gRPC client instance:**
```python
import ovmsclient

client = ovmsclient.make_grpc_client("localhost:9000")
```

**Create and send model status request:**
```python
model_status = client.get_model_status(model_name="model")

# Exemplary model_status:
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

# Exemplary model_metadata. Values for model:
# https://docs.openvino.ai/2022.2/omz_models_model_resnet_50_tf.html
#
#{
#   "model_version": 1,
#   "inputs": {
#       "map/TensorArrayStack/TensorArrayGatherV3": {
#           "shape": [1, 224, 224, 3],
#           "dtype": DT_FLOAT32  
#       }
#   },
#   "outputs": {
#       "softmax_tensor": {
#           "shape": [1, 1001],
#           "dtype": DT_FLOAT32  
#       }
#   }
#}
#
```

**Create and send predict request with binary input data:**
```python
# Assuming requesting model with inputs and outputs as in:
# https://docs.openvino.ai/2022.2/omz_models_model_resnet_50_tf.html

with open(<path_to_img>, 'rb') as f:
    img = f.read()
inputs = {"map/TensorArrayStack/TensorArrayGatherV3": img}
results = client.predict(inputs=inputs, model_name="model")

# Exemplary results:
#
# [[0.01, 0.03, 0.91, ... , 0.00021]]
#
```

For more details on `ovmsclient` see [API reference](https://github.com/openvinotoolkit/model_server/blob/releases/2022/3/client/python/ovmsclient/lib/docs/README.md)