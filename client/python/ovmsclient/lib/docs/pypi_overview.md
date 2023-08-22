# OpenVINO&trade; Model Server Client

OpenVINO&trade; Model Server Client package makes the interaction with the model server easy. It is very lightweight thanks to minimal number of included dependencies. The total size of the package, along with all dependencies is less than 100 MB.


The `ovmsclient` package works both with OpenVINO&trade; Model Server and TensorFlow Serving. It supports both gRPC and REST API calls: `Predict`, `GetModelMetadata` and `GetModelStatus`.


The `ovmsclient` can replace `tensorflow-serving-api` package with reduced footprint and simplified interface.


See [API reference](https://github.com/openvinotoolkit/model_server/blob/releases/2023/0/client/python/ovmsclient/lib/docs/README.md) for usage details.


## Usage example

```python
import ovmsclient

# Create connection to the model server
client = ovmsclient.make_grpc_client("localhost:9000")

# Get model metadata to learn about model inputs
model_metadata = client.get_model_metadata(model_name="model")

# If model has only one input, get its name
input_name = next(iter(model_metadata["inputs"]))

# Read the image file
with open("path/to/img.jpg", 'rb') as f:
    img = f.read()

# Place the data in a dict, along with model input name
inputs = {input_name: img}

# Run prediction and wait for the result
results = client.predict(inputs=inputs, model_name="model")

```

Learn more on `ovmsclient` [documentation site](https://github.com/openvinotoolkit/model_server/tree/releases/2023/0/client/python/ovmsclient/lib).