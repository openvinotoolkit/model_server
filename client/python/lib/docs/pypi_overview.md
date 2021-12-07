# OpenVINO&trade; Model Server Client

OpenVINO&trade; Model Server Client package contains a set of utilities to simplify interaction with the model server. To make the package as lightweight as possible, only the necessary dependencies have been included, making the total size of the package, along with all dependencies less than 100 MB.

The `ovmsclient` package works both with OpenVINO Model Server and TensorFlow Serving on: Predict, GetModelMetadata and GetModelStatus endpoints.

See [API documentation](https://github.com/openvinotoolkit/model_server/blob/main/client/python/lib/docs/README.md) for details on what the `ovmsclient` package provides.

## Usage

```python
import ovmsclient

# Create connection to the model server
client = ovmsclient.make_grpc_client("localhost:9000")

# Get model metadata to learn about model inputs
model_metadata = client.get_model_metadata(model_name="model")

# If model has only one input, get its name like that
input_name = next(iter(model_metadata["inputs"]))

# Read the image file
with open("path/to/img.jpg", 'rb') as f:
    img = f.read()

# Place the data in a dict, along with model input name
inputs = {input_name: img}

# Run prediction and wait for the result
results = client.predict(inputs=inputs, model_name="model")

```

See more [examples](https://github.com/openvinotoolkit/model_server/blob/main/client/python/samples) of how to use `ovmsclient`.