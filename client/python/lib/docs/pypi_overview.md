# OpenVINO&trade; Model Server Client

OpenVINO&trade; Model Server Client is a set of objects and methods designed to simplify user interaction with the instance of the model server. The package contains functions that hide API specific aspects, so user doesn't have to know about creating protos, preparing requests, parsing responses etc. and can focus on the application itself, rather than dealing with all the aspects of the interaction with the model server.

OVMS Client contains only the necessary dependencies, so the whole package is light. That makes it more friendly for deployments with restricted resources as well as for the use cases that require applications to scale well.

As OpenVINO Model Server API is compatible with TensorFlow Serving, it's possible to use `ovmsclient` with TensorFlow Serving instances on: Predict, GetModelMetadata and GetModelStatus endpoints.

See [API documentation](https://github.com/openvinotoolkit/model_server/blob/main/client/python/lib/docs/README.md) for details on what the `ovmsclient` package provides.


## Installation

**Note:** The package requires Python in version >= 3.6.

```
pip3 install ovmsclient
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
# https://docs.openvinotoolkit.org/latest/omz_models_model_resnet_50_tf.html
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
# https://docs.openvinotoolkit.org/latest/omz_models_model_resnet_50_tf.html

with open(<path_to_img>, 'rb') as f:
    img = f.read()
inputs = {"map/TensorArrayStack/TensorArrayGatherV3": img}
results = client.predict(inputs=inputs, model_name="model")

# Exemplary results:
#
# [[0.01, 0.03, 0.91, ... , 0.00021]]
#
```

For more details on `ovmsclient` see [API reference](https://github.com/openvinotoolkit/model_server/blob/main/client/python/lib/docs/README.md)