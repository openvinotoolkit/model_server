# OpenVINO&trade; Model Server Client Library

Model server client library is a set of objects and methods designed to simplify user interaction with the instance of the model server. The library contains functions that hide API specific aspects, so user doesn't have to know about creating protos, preparing requests, parsing responses etc. and can focus on the application itself, rather than dealing with all the aspects of the interaction with OVMS.



See [API documentation](docs/README.md) for details on what the library provides.


## Installation

The client library requires Python in version >= 3.6.

### Linux

Prerequisites:
 - Python 3.6 +
 - Python package [setuptools](https://pypi.org/project/setuptools/)
 - [protobuf-compiler](https://grpc.io/docs/protoc-installation/)

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
{
    "1": {
        "state": <model_version_state>, 
        "error_code": <error_code>, 
        "error_message": <error_message>
    }             
} 
```

**Create gRPC client instance:**
```python
import ovmsclient

config = {
   "address": "localhost", 
   "port": 9000
   }

grpc_client = ovmsclient.make_grpc_client(config)
```