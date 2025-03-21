# Python support in OpenVINO Model Server - quickstart {#ovms_docs_python_support_quickstart}

OpenVINO Model Server allows users to write custom processing nodes in Python, so they may have full control over what happens with the data reaching such node and what comes out of it.

In this quickstart you will create a servable with a single custom Python code that will expect a string and return the same string, but all in uppercase. 

Check out the [documentation](reference.md) to learn more about this feature.

To achieve that let's follow the steps:
1. Prepare Workspace
2. Write Python Code For The Server 
3. Prepare Graph Configuration File
4. Prepare Server Configuration File
5. Deploy OpenVINO Model Server
6. Create Client Application
7. Send Requests From The Client

### Step 1: Prepare Workspace

Let's have all the work done in a new `workspace` directory. Also create the following subdirectories:
- `workspace/models` (catalog that will be mounted to the deployment)
- `workspace/models/python_model` (catalog with Python servable specification)

You can do that in one go:

```bash
mkdir -p workspace/models/python_model && cd workspace
```

Since changing all the letters in the string to uppercase is a very basic example, the basic Python-enabled model server image without extra Python packages is sufficient. If you need some external modules, you need to add them to the image manually. For that simple use case, let's use publicly available `openvino/model_server:latest` image from Docker Hub.

You will also need a client module, so in your environment install a required dependency:
```bash
pip3 install tritonclient[grpc]
```

### Step 2: Write Python Code For The Server 

Let's start with the server side code. Your job is to implement an `OvmsPythonModel` class. Model Server expects it to have at least `execute` method implemented.

In basic configuration `execute` method is called every time model receives a request. The server reads inputs from that request and passes them to `execute` function as an `inputs` argument. 

`inputs` is a `list` of `pyovms.Tensor` objects. In this case you will have only one input so the code can start like this:

```python
input_data = inputs[0]
```

Now `input_data` is `pyovms.Tensor` object that holds the data and some metadata like shape and datatype. At this point you need to decide what kind of data you expect to receive here.

You will work on a string, so let's say you expect `input_data` to be UTF-8 encoded string. In that case you can create an instances of `bytes` from `input_data` and then decode it to an actual string:

```python
text = bytes(input_data).decode()
```

You also get to decide in what format you want to return the data. It makes sense to also return UTF-8 encoded string, so let's do that:

```python
output_data = text.upper().encode()
```

The outputs are expected to be a `list` of `pyovms.Tensor`, so you will need to pack the output to `pyovms.Tensor` with proper output name (in that case `uppercase`) and return it as a list. 

The complete code would look like this:

```python
from pyovms import Tensor

class OvmsPythonModel:

    def execute(self, inputs: list):
        input_data = inputs[0]
        text = bytes(input_data).decode()
        output_data = text.upper().encode()
        return [Tensor("uppercase", output_data)]
 ```

Let's create `model.py` file with that code and save it to `workspace/models/python_model`

```bash
echo '
from pyovms import Tensor

class OvmsPythonModel:

    def execute(self, inputs: list):
        input_data = inputs[0]
        text = bytes(input_data).decode()
        output_data = text.upper().encode()
        return [Tensor("uppercase", output_data)]
' >> models/python_model/model.py
```

### Step 3: Prepare Graph Configuration File

Python logic execution in OpenVINO Model Server is supported via MediaPipe graphs. That means you need to prepare graph definition for your processing flow. In that case, a graph with just one node - Python node - is enough. Let's create appropriate `graph.pbtxt` file in your `workspace/models/python_model` catalog:

```bash
echo '
input_stream: "OVMS_PY_TENSOR:text"
output_stream: "OVMS_PY_TENSOR:uppercase"

node {
  name: "pythonNode"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:text"
  output_stream: "OUTPUT:uppercase"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/models/python_model/model.py"
    }
  }
}' >> models/python_model/graph.pbtxt
```

Above configuration file creates a graph with a single Python node that uses `PythonExecutorCalculator`, sets inputs and outputs and provides your Python code location in `handler_path`. 

`input_stream` and `output_stream` in the first two lines define graph inputs and outputs. The names to the **right** of `:` are names to be used in request and response. In this case: 
- **input**: "text"
- **output**: "uppercase"

You can also see `input_stream` and `output_stream` on the node level. Those refer to naming in the `execute` method code. Notice how in the previous step, in `execute` implementation, you name the output tensor - "uppercase". 

In that case the names of the streams both on the graph and on the node level are exactly the same, which means that a graph input is also a node input and a node output is also a graph output.

The `input_side_packet` value is an internal field used by the model server to share resources between graph instances - do not change it. 

### Step 4: Prepare Server Configuration File

Last piece of configuration would be the model server configuration file. 
Create `config.json` with the following content in `workspace/models`:

```bash
echo '
{
    "model_config_list": [],
    "mediapipe_config_list": [
    {
        "name":"python_model",
        "graph_path":"/models/python_model/graph.pbtxt"
    }
    ]
}
' >> models/config.json
```

This tells OpenVINO Model Server to to serve the graph under given name `python_model`.

### Step 5: Deploy OpenVINO Model Server

Before running the server let's check if all files required for deployment are in place. Check the contents of `workspace/models` catalog as it will be mounted to the container:
```bash
tree models
models
├── config.json
└── python_model
    ├── graph.pbtxt
    └── model.py
```

Now let's run the server:
```bash
docker run -it --rm -p 9000:9000 -v $PWD/models:/models openvino/model_server:latest --config_path /models/config.json --port 9000
```

### Step 6: Create Client Application

Now that the Python model is deployed, you can focus on the other end - the client application. When writing the client keep in mind how the server side code looks like as they must be complementary.

First let's connect to the server hosted on `localhost` with gRPC interface available on port `9000`:

```python
import tritonclient.grpc as grpcclient
client = grpcclient.InferenceServerClient("localhost:9000")
```

You will send a string, so let's create one and encode it to UTF-8, because that's what the server side code expects:

```python
data = "Make this text uppercase.".encode()
```

Now let's pack that data into a gRPC structure that will be sent to the server:

```python
infer_input = grpcclient.InferInput("text", [len(data)], "BYTES")
infer_input._raw_content = data
```

You've created InferInput object that will correspond to the graph input with the name "text", shape [len(data)] - where len(data) is the number of encoded bytes - and datatype "BYTES". The data itself has been written to a raw_content field. All of these values can be accessed on the server side.

The last part would be to send this data to the server:

```python
results = client.infer("python_model", [infer_input])
print(results.as_numpy("uppercase").tobytes().decode())
```

That part will pack `infer_input` into a request and send it to the servable called `uppercase_model`. 

Server is expected to respond with an output containing UTF-8 encoded string, so in the second line you read it, decode it to an actual string and print it.


Let's save the entire code to `client.py` file inside `workspace`:

```bash
echo '
import tritonclient.grpc as grpcclient
client = grpcclient.InferenceServerClient("localhost:9000")
data = "Make this text uppercase.".encode()
infer_input = grpcclient.InferInput("text", [len(data)], "BYTES")
infer_input._raw_content = data
results = client.infer("python_model", [infer_input])
print(results.as_numpy("uppercase").tobytes().decode())
' >> client.py 
```

### Step 7: Send Requests From The Client

Once you have model server up and running, let's send a text: `"Make this text uppercase."`. 

Simply run your `client.py` from the `workspace` catalog and see the results:

```bash
python client.py
MAKE THIS TEXT UPPERCASE.
```

