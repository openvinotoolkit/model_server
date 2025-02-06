# Python Execution in OpenVINO Model Server {#ovms_docs_python_support_reference}

## Introduction

 Starting with version 2023.3, OpenVINO Model Server supports execution of custom Python code. Such code can execute simple pre- or post-processing as well as complex tasks like image or text generation.

 Python execution is enabled via [MediaPipe](../mediapipe.md) by the built-in [`PythonExecutorCalculator`](#pythonexecutorcalculator) that allows creating graph nodes to execute Python code. Python nodes can be used as standalone servables (single node graphs) or be part of larger MediaPipe graphs.

 Check out the [quickstart guide](quickstart.md) for a simple example that shows how to use this feature.

 Check out [Generative AI demos](../../demos/README.md#check-out-new-generative-ai-demos) for real life use cases.

 ## Building Docker Image

The publicly available `openvino/model_server` image on Docker Hub supports Python, but does not come with external modules installed. If Python is all you need then you can use the public image without modification. Otherwise, you will need to extend the public image with additional layers that install any modules required for your Python code to run. For example, let's say your code requires numpy.
In that case, your Dockerfile may look like this:

```dockerfile
FROM openvino/model_server:latest
USER root
ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python
RUN apt update && apt install -y python3-pip git
RUN pip3 install numpy
ENTRYPOINT [ `/ovms/bin/ovms` ]
```

You can also modify `requirements.txt` from our [python demos](https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos) and from repository top level directory run `make python_image`

## `OvmsPythonModel` class

When deploying a Python node, the Model Server expects a Python file with an `OvmsPythonModel` class implemented:

```python
class OvmsPythonModel:

    def initialize(self, kwargs):
        """
        `initialize` is called when model server loads graph definition.
        It allows to initialize and maintain state between subsequent execute() calls
        and even graph instances. For gRPC unary, graphs are recreated per request.
        For gRPC streaming, there can be multiple graph instances existing at the same time.
        OvmsPythonModel object is initialized with this method and then shared between all graph instances.
        Implementing this function is optional.

        Parameters:
        -----------
        kwargs : dict
            Available arguments:
            * node_name: string
                Name of the node in the graph
            * input_names: list of strings
                List of input stream names defined for the node in graph
                configuration
            * output_names: list of strings
                List of output stream names defined for the node in graph
                configuration
        -----------
        """
        print("Running initialize...")

    def execute(self, inputs):
        """
        `execute` is called in `Process` method of PythonExecutorCalculator
        which in turn is called by the MediaPipe framework. How MediaPipe
        calls the `Process` method for the node depends on the configuration
        and the two configurations supported by PythonExecutorCalculator are:

        * Regular: `execute` is called with a set of inputs and returns a set of outputs.
        For unary endpoints it's the only possible configuration.

        * Generative: `execute` is called with a set of inputs and returns a generator.
        The generator is then called multiple times with no additional input data and produces
        multiple sets of outputs over time. Works only with streaming endpoints.

        Implementing this function is required.

        Parameters:
        -----------
        * inputs: list of pyovms.Tensor
        -----------

        Returns: list of pyovms.Tensor or generator
        """
        ...
        return outputs

    def finalize(self):
        """
        `finalize` is called when model server unloads graph definition.
        It allows to perform any cleanup actions before the graph definition
        is removed. Implementing this function is optional.
        """
        print("Running finalize...")
```

### `initialize`

`initialize` is called when model server loads graph definition. It allows to initialize and maintain state between subsequent `execute` calls and even graph instances.

For unary endpoint, graphs are recreated per request.

For gRPC streaming, there can be multiple graph instances existing at the same time.

`OvmsPythonModel` object is initialized with this method and then shared between all graph instances.

#### Parameters and return value

`initialize` is called with `kwargs` parameter which is a dictionary.
`kwargs` contain information from [node configuration](#pythonexecutorcalculator). Considering a sample:

```pbtxt
node {
  name: <NODE_NAME>
  ...
  input_stream: "<INPUT_TAG>:<INPUT_NAME>"
  input_stream: "<INPUT_TAG>:<INPUT_NAME>"
  ...
  output_stream: "<OUTPUT_TAG>:<OUTPUT_NAME>"
  output_stream: "<OUTPUT_TAG>:<OUTPUT_NAME>"
  ...
}
```

All keys are strings. Available keys and values:

| Key           | Value type | Description |
| ------------- |:-----------| :-----------|
| node_name     | string | Name of the node in the graph. `<NODE_NAME>` in the sample above |
| input_names   | list of strings | List of `<INPUT_NAME>` from all input streams in the sample above |
| outputs_names | list of strings | List of `<OUTPUT_NAME>` from all output streams in the sample above |
| base_path     | string | Path to the folder containing handler script |

`initialize` is not expected to return any value.

#### Error handling

Signaling that something went wrong should be done by throwing an exception.
When model server catches exception from `initialize` it cleans up all Python resources in the graph (including those belonging to the correctly loaded nodes) and sets the whole graph in unavailable state.

**Note**: Run Model Server with `--log_level DEBUG` parameter to get information about errors in the server logs.

*Implementing this function is optional*

### `execute`

`execute` is called in `Process` method of `PythonExecutorCalculator` which in turn is called by MediaPipe framework. How MediaPipe calls `Process` for the node depends on the configuration and the two configurations supported by `PythonExecutorCalculator` are:

#### Regular

`execute` is called with a set of inputs and returns a set of outputs. For unary endpoints it's the only possible configuration. On the implementation side, to use that mode, `execute` should `return` outputs.

```python
def execute(self, inputs):
    ...
    return outputs
```

More information along with the configuration aspect described can be found in [execution modes](#execution-modes) section.

#### Generative

`execute` is called with a set of inputs and returns a [generator](https://wiki.python.org/moin/Generators). The generator is then called multiple times with no additional input data and produces multiple sets of outputs over time. Works only with streaming endpoints. On the implementation side, to use that mode, `execute` should `yield` outputs.

```python
def execute(self, inputs):
    # For single set on inputs generate 10 sets of outputs
    for _ in range(10):
        ...
        yield outputs
```

More information along with the configuration aspect described can be found in [execution modes](#execution-modes) section.

#### Parameters and return value

`execute` is called with `inputs` parameter which is a `list of pyovms.Tensor`.

Depending on the mode it should return:

- For regular mode: `list of pyovms.Tensor`
- For generative mode: `generator` that generates `list of pyovms.Tensor`

So depending on the mode `execute` must always either `return` or `yield` a `list of pyovms.Tensor`

**Returning multiple Python outputs from the graph**

Note that this method returns outputs as a list, but since each output is a separate packet in MediaPipe flow, they do not arrive together to their destination. If the node outputs are also outputs from the graph the behavior differs depending on the kind of endpoint used:

- For unary endpoints model server gathers all outputs from the graph and sends them all together in a single response

- For streaming endpoints model server packs output and sends it in the response as soon as it arrives. It means that if `execute` returns a list of `X` outputs, the client will receive those outputs in `X` separate responses. The outputs can then be [gathered using timestamp](#outputs-synchronization-in-grpc-streaming) that can be found in received responses.

#### Error handling

Signaling that something went wrong should be done by throwing an exception.
The exception is caught by the `PythonExecutorCalculator` which logs it and returns non-OK status.
Model Server then reads that status and sets graph in an error state. Then it closes all graph's input streams and waits until in-progress actions are finished. Once it's done the graph gets removed.

This behavior has different effect on the client depending on the kind of endpoint used - unary or streaming:

- **Unary**

  With unary endpoint a graph is created, executed and destroyed for every request. If `execute` encounters an error, model server logs it and sends error message in response immediately.

- **Streaming**

  With streaming endpoint a graph is created for the first request in the stream and then reused by all subsequent requests.

  If `execute` encounters an error on the first request (for example the Python code doesn't work as expected), model server logs it  and sends error message in response immediately. The graph gets destroyed.

  If `execute` encounters an error on one of the subsequent requests (for example wrong data has been received), model server logs it and MediaPipe sets error in the graph, but the client won't receive error message until it sends another request. When the next request is read from the stream, model server checks if graph has an error, destroys it and sends response to the client. Rework of that behavior, so that error is being sent immediately is planned.

  As of now, graphs are not recoverable, so if `execute`  encounters an error the stream gets closed and you need to create a new one.

**Note**: Run Model Server with `--log_level DEBUG` parameter to get information about errors in the server logs.

**Implementing this function is required.**

### `finalize`

`finalize` is called when model server unloads graph definition. It allows to perform any cleanup actions before the graph is removed.

#### Parameters and return value

`finalize` does not have any parameters and is not expected to return any value.

#### Error handling

Signaling that something went wrong should be done by throwing an exception.
When model server catches exception from `finalize` it logs it and proceeds with the unload.

**Note**: Run Model Server with `--log_level DEBUG` parameter to get information about errors in the server logs.

*Implementing this function is optional.*

## Python Tensor

`PythonExecutorCalculator` operates on a dedicated `Tensor` class that wraps the data along with some additional information like name, shape or datatype. Objects of that class are passed to `execute` method as inputs and returned as output. They are also wrapped and exchanged between nodes in the graph and between graph and model server core.

This `Tensor` class is a C++ class with a Python binding that implements Python [Buffer Protocol](https://docs.python.org/3/c-api/buffer.html#buffer-protocol). It can be found in a built-in module `pyovms`.

### Accessing Tensor Contents

`pyovms.Tensor` attributes:

| Name           | Type | Description |
| ------------- |:-----------| :-----------|
| name     | string | Name of the string that also associates it with input or output stream of the node |
| shape   | tuple | Tuple of numbers defining the shape of the tensor |
| datatype | string | Type of elements in the buffer. |
| data | memoryview | Memoryview of the underlying data buffer |
| size | number | Size of data buffer in bytes |

*Note*: `datatype` attribute is not part of buffer protocol implementation.
Buffer protocol uses `format` value that uses [struct format characters](https://docs.python.org/3/library/struct.html#format-characters). It can be read from `data` memoryview.
There's a mapping between those two - see [datatype considerations](#datatype-considerations).

As `pyovms.Tensor` implements buffer protocol it can be converted to another types that also implement buffer protocol:

```python
def execute(self, inputs):
    input_tensor_bytes = bytes(inputs[0])
    ...
    import numpy as np
    input_tensor_ndarray = np.array(inputs[1])
    ...
```
### Creating Output Tensors

Inputs will be provided to the `execute` function, but outputs must be prepared by the user. Output objects can be created using `pyovms.Tensor` class constructor:

`Tensor(name, data, shape=None, datatype=None)`

- `name`: a string that associates Tensor data with specific name. This name is also used by `PythonExecutorCalculator` to push data to the correct output stream in the node. More about it in [node configuration section](#input-and-output-streams-in-python-code).

- `data`: an object that implements Python [Buffer Protocol](https://docs.python.org/3/c-api/buffer.html#buffer-protocol). This could be an instance of some built-in type like `bytes` or types from external modules like `numpy.ndarray`.

- `shape` (*optional*): a tuple or list defining the shape of the data. This value is directly assigned to `shape` attribute of the `Tensor`. By default, `shape` attribute is inherited from the `data` object. Providing `shape` to the constructor will override inherited value, so use it only if you know what you are doing.

- `datatype` (*optional*): a string defining the type of the data. This value is directly assigned to `datatype` attribute of the `Tensor`. By default, `datatype` attribute is inherited from the `data` object. Providing `datatype` to the constructor will override inherited value, so use it only if you know what you are doing.

**Note**: `shape` and `datatype` arguments do not modify internal structure of the data - there are no reshapes and type conversions. They only override `Tensor.shape` and `Tensor.datatype` attributes, so the user can provide custom context to the next node or server response. It means they can be completely detached from the data buffer properties and it's user's responsibility to correctly interpret these attributes while reading the `Tensor` in the next node or the server response on the client side.

```python
import numpy as np
from pyovms import Tensor

class OvmsPythonModel:
    def execute(self, inputs):
        # Create Tensor called my_output1 with encoded text
        output1 = Tensor("my_output1", "some text".encode())

        # Create Tensor called my_output with batch of string in numpy array
        # with overriding tensor datatype to match numpy array dtype
        npy_arr = np.array(["my", "batched", "string", "output"])
        output2 = Tensor("my_output2", npy_arr, datatype=np_arr.dtype.str)

        # A list of Tensors is expected, even if there's only one output
        return [output1, output2]
```

As `Tensor` gets created from another type it adapts all fields required by the buffer protocol as its own.
Depending on how `Tensor` is created `shape` or `datatype` may be overridden.
If they are not provided `Tensor` will adapt another buffer `shape` as it's own and will map it's `format` to a `datatype`. Learn more in [datatype considerations](#datatype-considerations) section.

If the node is connected to another Python node, then Tensors pushed to the output of this node, are inputs of another node.

### Datatype considerations

There are two places where `pyovms.Tensor` objects are created and accessed:
- in `execute` method of `OvmsPythonModel` class
- in model server core during serialization and deserialization if Python node inputs or outputs as also graph inputs or outputs

Model Server receives requests and sends responses on interface via KServe API which defines [expected data types for tensors](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data-types).
On the other hand Python [Buffer Protocol](https://docs.python.org/3/c-api/buffer.html#buffer-protocol) requires `format` to be specified as [struct format characters](https://docs.python.org/3/library/struct.html#format-characters).

In order to let users work with KServe types without enforcing the usage of struct format characters on the client side, model server attempts to do the mapping as follows when creating `Tensor` objects from the request:

| KServe Type   | Format Character |
| :------------ |:----------------:|
|`BOOL`         | `?`              |
|`UINT8`        | `B`              |
|`UINT16`       | `H`              |
|`UINT32`       | `I`              |
|`UINT64`       | `Q`              |
|`INT8`         | `b`              |
|`INT16`        | `h`              |
|`INT32`        | `i`              |
|`INT64`        | `q`              |
|`FP16`         | `e`              |
|`FP32`         | `f`              |
|`FP64`         | `d`              |


The same mapping is applied the other way around when creating `Tensor` from another Python object in `execute` method (unless `datatype` argument is provided to the [constructor](#creating-output-tensors) ).

`Tensor` object always holds both values in `Tensor.datatype` and `Tensor.data.format` attributes so they can be used in deserialization and serialization, but also in another node in the graph.

In some cases, users may work with more complex types that are not listed above and model server also allows that.

#### BYTES datatype
If `datatype` "BYTES" is specified and data is located in bytes_contents field of input(for gRPC) or in JSON body(for REST) OVMS converts it to `pyovms.Tensor` buffer according to the format where every input is preceded by four bytes of its size.

For example this gRPC request:
 bytes_content: [<240 byte element>, <1024 byte element>, <567 byte element>]

would be converted to this pyovms.Tensor.data contents:
| 240 |   < first element>  | 1024 |   <second element> | 567 | <third element> |

#### Custom types

The `datatype` field in the tensor is a `string` and model server will not reject datatype that is not among above KServe types. If some custom type is defined in the request and server cannot map it to a format character it will translate it to `B` treating it as a 1-D raw binary buffer. For consistency the shape of the underlying buffer in that case will also differ from the shape defined in the request. Let's see it on an example:

1. Model Server receives request with the following input:
    * datatype: "my_string"
    * shape: (3,)
    * data: "null0terminated0string0" string encoded in UTF-8

2. Model Server creates `pyovms.Tensor` with:
    * Tensor.datatype: "my_string"
    * Tensor.shape: (3,)
    * Tensor.data.format: "B"
    * Tensor.data.shape: (23,)

In `execute` user has access to both information from the request as well as how the internal buffer looks like.

`pyovms.Tensor` objects produced inside `execute` inherit most of the fields from the objects they are created from and by default tensor will try to map buffer protocol `format` to `datatype` according to the mapping mentioned before.

If it fails, the `datatype` is set to `format`, so that if such tensor is the output tensor of the graph, client receives the most valuable information about the type of output data.

In case this approach is insufficient, user can manually set `datatype` attribute to more suitable one, by providing optional `datatype` argument to the `Tensor` [constructor](#creating-output-tensors).

## Configuration and deployment

Python is enabled via [MediaPipe](../mediapipe.md) by built-in `PythonExecutorCalculator`, therefore, in order to execute Python code in OVMS you need to create a graph with a node that uses this calculator.

The way the graph is configured has a huge impact on the whole deployment. It defines things like:
- inputs and outputs of the graph
- inputs and outputs of each node in the graph
- connections between the nodes
- graph and nodes options
- input stream handlers (defines conditions that must be met to launch `Process` in the node)

### PythonExecutorCalculator

Main part of the configuration is the node setting. Python nodes should use `PythonExecutorCalculator` and **must** be named. See a basic example:

```pbtxt
node {
  name: "python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:input"
  output_stream: "OUTPUT:output"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/model.py"
    }
  }
}
```

Let's break it down:

- `name`: the name by which the node will be identified in the model server. Every Python node in a graph must have a unique name.

- `calculator`: indicates the calculator to be used in the node. Must be `PythonExecutorCalculator`.

- `input_side_packet`: a shared data passed from the model server to the Python nodes. It allows to share `OvmsPythonModel` state between multiple graph instances. Must be `PYTHON_NODE_RESOURCES:py`.

- `input_stream`: defines input in form `[TAG]:[NAME]`. MediaPipe allows configurations with indexes i.e. `[TAG]:[INDEX]:[NAME]`, but `PythonExecutorCalculator` ignores it.

- `output_stream`: defines output in form `[TAG]:[NAME]`. MediaPipe allows configurations with indexes i.e. `[TAG]:[INDEX]:[NAME]`, but `PythonExecutorCalculator` ignores it.

- `handler_path`: the only one options so far in `PythonExecutorCalculator`. It's a path to the Python file with `OvmsPythonModel` implementation.

### Input and output streams in Python code

How node input and output streams are configured has direct impact on the names of `pyovms.Tensor` objects in `execute` method of `OvmsPythonModel`. In previous simple configuration there are:
```pbtxt
input_stream: "INPUT:input"
output_stream: "OUTPUT:output"
```

Both input and output streams are constructed as `[TAG]:[NAME]`.
So in this example there's:
- input with tag `INPUT` and name `input`
- output with tag `OUTPUT` and name `output`

In the Python code you should always refer to the `[NAME]` part.
So inside `execute` there would be:

```python
from pyovms import Tensor

class OvmsPythonModel:
    def execute(self, inputs):
        my_input = inputs[0]
        my_input.name == "input" # true
        my_output = Tensor("output", "some text".encode())
        return [my_output]
```

#### Access inputs via index

In basic configurations, when `execute` runs with all expected inputs the order of `Tensors` in `inputs` list is not random. When `PythonExecutorCalculator` iterates through input streams to create `Tensors`, the streams are sorted by their tags. That knowledge can be useful while writing `execute` method to directly access data from particular streams. See an example:

```pbtxt
node {
  name: "python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "B:b"
  input_stream: "A:a"
  input_stream: "C:c"
  output_stream: "OUTPUT:output"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/model.py"
    }
  }
}
```

In that case, inputs can be access like this:
```python
from pyovms import Tensor

class OvmsPythonModel:
    def execute(self, inputs):
        a = inputs[0] # Tensor with name "a" from input stream with tag "A"
        b = inputs[1] # Tensor with name "b" from input stream with tag "B"
        c = inputs[2] # Tensor with name "c" from input stream with tag "C"
        ...
```

**Note**: Node configuration and `execute` implementation should always match. For example if the node is configured to work with [incomplete inputs](#incomplete-inputs), then accessing `Tensors` via index will not be useful.

### Graph input and output streams

So far only node input and output streams have been mentioned, but the configuration also requires defining graph's input and output streams.
The rules are very similar to how it works on the node level, so the streams are described in form: `[TAG]:[NAME]`, but there's more to it.

On graph level the `[TAG]` helps model server in deserialization and serialization by providing information about the object type expected in the stream. Model server reads the tag and expects it to start with one of predefined prefixes. If graph input stream is connected to Python node the tag should begin with `OVMS_PY_TENSOR`, which tells the server that it should deserialize input from the request to `pyovms.Tensor` object.

There can't be two or more the same tags among the input streams as well as there can't be two or more the same tags among the output streams. In such cases, prefix must be followed by some unique string.

```pbtxt
input_stream: "OVMS_PY_TENSOR_IMAGE:image"
input_stream: "OVMS_PY_TENSOR_TEXT:text"
output_stream: "OVMS_PY_TENSOR:output"
```

**Note**: The same rule applies to **node** input and output streams.

When it comes to the `[NAME]` part, it is used to connect graph inputs and output with the nodes. They are also the input and output names in server requests and responses.

### Multiple nodes
Here is what you should know if you want to have multiple Python nodes in the same graph:

- Every Python node must have a unique name in graph scope
- Every Python node has it's own instance of `OvmsPythonModel` that is not shared even if two nodes have identical `handler_path`
- Nodes based on `PythonExecutorCalculator` can be connected directly without need for converters
- Nodes may reuse the same Python file, but every Python file used by the server must have a unique name, otherwise some nodes might not work as expected.
For example: `/ovms/workspace1/model.py` and `/ovms/workspace2/model.py` will result in only one `model.py` effectively loaded (this is supposed to be changed in the future versions).

### Basic example
Let's see a complete example of the configuration with three Python nodes set in sequence:

```pbtxt
input_stream: "OVMS_PY_TENSOR:first_number"
output_stream: "OVMS_PY_TENSOR:last_number"

node {
  name: "first_python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:first_number"
  output_stream: "OUTPUT:second_number"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/incrementer.py"
    }
  }
}

node {
  name: "second_python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:second_number"
  output_stream: "OUTPUT:third_number"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/incrementer.py"
    }
  }
}

node {
  name: "third_python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:third_number"
  output_stream: "OUTPUT:last_number"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/incrementer.py"
    }
  }
}
```

In that example client will send an input called `first_number` and receive output called `last_number`. Since user has access to input and output names in the Python code, the code for incrementation can be generic and reused in all nodes.

`incrementer.py`
```python
from pyovms import Tensor

def increment(input):
    # Some code for input incrementation
    ...
    return output

class OvmsPythonModel:
    # Assuming this code is used with nodes
    # that have single input and single output

    def initialize(self, kwargs):
        self.output_name = kwargs["output_names"][0]

    def execute(self, inputs):
        my_input = inputs[0]
        my_output = Tensor(self.output_name, increment(my_input))
        return [my_output]
```

### Model Server configuration file

Once Python code and the `pbtxt` file with graph configuration is ready the model server configuration is very simple and could look like this:

```json
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name":"python_graph",
            "graph_path":"/ovms/workspace/graph.pbtxt"
        }
    ]
}
```
Where `name` defines the name of the graph and `graph_path` contains the path to graph configuration file.

## Client side considerations

### Inference API and available endpoints

Since Python execution is supported via MediaPipe serving flow, it inherits it's enhancements and limitations. First thing to note is that MediaPipe graphs are available [**only via KServe API**](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md)

From the client perspective model server serves a graph and user interacts with a graph. Single node in the graph cannot be accessed from the outside.

For a graph client can:

- request status
- request metadata
- request inference

Learn more about how [MediaPipe flow works in OpenVINO Model Server](../mediapipe.md)

For inference, data can be send both via [gRPC API](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#grpc) and [KServe API](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#httprest)(only for unary calls). If the graph has a `OvmsPyTensor` output stream, then the data in the KServe response can be found in `raw_output_contents` field (even if data in the request has been placed in `InferTensorContents`).

The data passed in the request is accessible in `execute` method of the node connected to graph input via `data` attribute of [`pyovms.Tensor`](#python-tensor) object.
For data of type BYTES send in bytes_contents field of input(for gRPC) or in JSON body(for REST) OVMS converts it to `pyovms.Tensor` buffer according to the format where every input is preceded by four bytes of its size.

Inputs and outputs also define `shape` and `datatype` parameters. Those values are also accessible in `pyovms.Tensor`. For outputs, `datatype` and `shape` are by default read from the underlying buffer, but it is possible to overwrite them (see [`pyovms.Tensor constructor`](#creating-output-tensors). If you specify `datatype` as `BYTES` in your requests, make sure to review [datatype considerations](#datatype-considerations), since this type is treated differently than the others.

Let's see it on an example:

```python
# client.py

import tritonclient.grpc as grpcclient
...
client = grpcclient.InferenceServerClient("localhost:9000")
inputs = []
with open("image_path", 'rb') as f:
    image_data = f.read()
image_input = grpcclient.InferInput("image", [len(image_data)], "BYTES")
image_input._raw_content = image_data

text_encoded = "some text".encode()
text_input = grpcclient.InferInput("text", [len(text_encoded)], "BYTES")
text_input._raw_content = text_encoded

numpy_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
numpy_input = grpcclient.InferInput("numpy", numpy_array.shape, "FP32")
numpy_input.set_data_from_numpy(numpy_array)

results = client.infer("model_name", [image_input, text_input, numpy_input])
```

```python
# model.py

from pyovms import Tensor
from PIL import Image
import io
import numpy as np
...
class OvmsPythonModel:

    def execute(self, inputs):
        # Read inputs
        image_input = inputs[0]
        print(image_input.shape) # (<image_binary_size>, )
        print(image_input.datatype) # "BYTES"

        text_input = inputs[1]
        print(text_input.shape) # (<string_binary_size>, )
        print(text_input.datatype) # "BYTES"

        numpy_input = inputs[2]
        print(text_input.shape) # (2, 3)
        print(text_input.datatype) # "FP32"

        # Convert pyovms.Tensor objects to more useful formats

        # Pillow Image created from image bytes
        image = Image.open(io.BytesIO(bytes(image_data)))
        # Python string "some text"
        text = bytes(text_input).decode()
        # Numpy array with shape (2, 3) and dtype float32
        ndarray = np.array(numpy_input)
        ...

```
### Timestamping

Mediapipe graph works with packets and every packet has its timestamp. The timestamps of packets on all streams (both input and output) must be ascending.

When requesting inference, user can decide to use automatic timestamping, or send timestamps themself along with the request as `OVMS_MP_TIMESTAMP` parameter. Learn more about [timestamping](../../docs/streaming_endpoints.md#timestamping)

When it comes to Python node `PythonExecutorCalculator`:
- for [regular execution mode](#regular-mode) simply propagates timestamp i.e. uses input timestamp as output timestamp.
- for [generative execution mode](#generative-mode) it saves timestamp of the input and sends first set of outputs downstream with this timestamp. Then timestamp gets incremented with each generation, so next sets of output packages have ascending timestamp.

**Multiple generation cycles on a single graph instance**

Keep in mind that node keeps the timestamp and overwrites it every time new input arrives. It means that if you want to run multiple generation cycles on a single graph instance you **must** use manual timestamping - next request timestamp must be larger than the one received in the last response.

#### Outputs synchronization in gRPC streaming

Timestamping has a crucial role when synchronizing packets from different streams both on the inputs and outputs as well as inside the graph. MediaPipe provides outputs of the graph to the model server and what happens next depends on what endpoint is used:

- on gRPC unary endpoints server waits for the packets from all required outputs and sends them in a single response.
- on gRPC streaming endpoints server serializes output packets as soon as they arrive and sends them back in separate responses.

It means that if you have a graph that has two or more outputs and use gRPC streaming endpoint you will have to take care of gathering the outputs. You can do that using `OVMS_MP_TIMESTAMP`.

```python
timestamp = result.get_response().parameters["OVMS_MP_TIMESTAMP"].int64_param
```

## Advanced Configuration

### Execution modes

Python nodes can be configured to run in two execution modes - regular and generative.

In regular execution mode the node produces one set of outputs per one set of inputs. It works via both gRPC/REST unary and gRPC streaming endpoints and is a common mode for use cases like computer vision.

In generative execution mode the node produces multiple sets of outputs over time per single set of inputs. It works only via gRPC streaming endpoints and is useful for use cases where total processing time is big and you want to return some intermediate results before the execution is completed. That mode is well suited to Large Language Models to serve them in a more interactive manner.

Depending on which mode is used, both the Python code and graph configuration must be in line.

#### Regular mode

When using regular mode, the `execute` method in [`OvmsPythonModel`](#ovmspythonmodel-class) class must `return` value.

```python
from pyovms import Tensor
...
  def execute(self, inputs):
        ...
        my_output = Tensor("output", data)
        return [my_output]
```

When `execute` returns, the [`PythonExecutorCalculator`](#pythonexecutorcalculator) grabs the outputs and pushes them down the graph. Node `Process` method is called once per inputs set. Such implementation can be paired with basic graph setting, like:

```pbtxt
node {
  name: "python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:input"
  output_stream: "OUTPUT:output"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/model.py"
    }
  }
}
```

#### Generative mode

When using generative mode, the `execute` method in [`OvmsPythonModel`](#ovmspythonmodel-class) class must `yield` value.

```python
from pyovms import Tensor
...
  def execute(self, inputs):
        ...
        for data in data_stream:
          my_output = Tensor("output", data)
          yield [my_output]
```

When `execute` yields, the [`PythonExecutorCalculator`](#pythonexecutorcalculator) saves the generator. Then it repeatedly calls it until it reaches the end of generated sequence. Node `Process` method is called multiple times per single inputs set. To trigger such behavior a specific graph configuration is needed. See below:

```pbtxt
node {
  name: "python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:input"
  input_stream: "LOOPBACK:loopback"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
  output_stream: "OUTPUT:output"
  output_stream: "LOOPBACK:loopback"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/model.py"
    }
  }
}
```

Apart from basic configuration present also in regular mode, this graph contains some additional content. Let's review it.

1. `LOOPBACK` input and output stream

    ```
    input_stream: "LOOPBACK:loopback"
    ...
    output_stream: "LOOPBACK:loopback"
    ```

    This set of additional input and output stream enables internal cycle inside the node. It is used to trigger `Process` calls without any incoming packets and therefore call the generator without new data. The value in both input and output stream must be exactly the same and the `PythonExecutorCalculator` always expects the tag to be `LOOPBACK`.

    `LOOPBACK` input is not passed to `execute` method and user does not interact with it in any way.

2. Back Edge Annotation
    ```
    input_stream_info: {
      tag_index: 'LOOPBACK:0',
      back_edge: true
    }
    ```

    This part says that the input stream with tag `LOOPBACK` and index `0` is used to create a cycle. If there are more than one index for `LOOPBACK` tag, the `PythonExecutorCalculator` will ignore it.

3. `SyncSetInputStreamHandler`

    ```
    input_stream_handler {
      input_stream_handler: "SyncSetInputStreamHandler",
      options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
          sync_set {
            tag_index: "LOOPBACK:0"
          }
        }
      }
    }
    ```

    In regular configuration `DefaultInputStreamHandler` is used by default, but for generative mode it's not sufficient. When default handler is defined, node waits for all input streams before calling `Process`. In generative mode `Process` should be called once for data coming from the graph and then multiple times only by receiving signal on `LOOPBACK`, but inputs from a graph and `LOOPBACK` will never be present at the same time.

    For generative mode to work, inputs from the graph and `LOOPBACK` must be decoupled, meaning `Process` can be called with a set of inputs from the graph, but also with just `LOOPBACK`. It can be achieved via `SyncSetInputStreamHandler`. Above configuration sample creates a set with `LOOPBACK`, which also, implicitly creates another set, with all remaining inputs.
    Effectively there are two sets that do not depend on each other:
    - `LOOPBACK`
    - ... every other input specified by the user.

It's recommended not to reuse the same graph instance when the cycle is finished.
Instead, if you want to generate for new data, create new gRPC stream.

For working configurations and code samples see the [demos](../../demos/README.md#check-out-new-generative-ai-demos).

### Incomplete inputs

There are usecases when firing `Process` with only a subset of inputs defined in node configuration is desired. By default, node waits for all inputs with the same timestamp and launches `Process` once they're all available. Such behavior is implemented by the `DefaultInputStreamHandler` which is used by default.
To configure the node to launch `Process` with only a subset of inputs you should use a different input stream handler for different [input policy](https://developers.google.com/mediapipe/framework/framework_concepts/synchronization#input_policies).

Such configuration is used in [generative execution mode](#generative-mode), but let's see another example:

```pbtxt
node {
  name: "python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT1:labels"
  input_stream: "INPUT2:image"
  input_stream_handler {
    input_stream_handler: "ImmediateInputStreamHandler",
  }
  output_stream: "OUTPUT:result"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/model.py"
    }
  }
}
```

Node configured with `ImmediateInputStreamHandler` will launch `Process` when any input arrives (no synchronization at all). Such configuration must be in line with the `OvmsPythonModel` class implementation. For example:

```python
from pyovms import Tensor

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        self.model = load_model(...)
        self.labels = []

    def execute(self, inputs: list):
        outputs = []
        for input in inputs:
            if input.name == "labels":
                self.labels = prepare_new_labels(input)
            else: # the only other name is "image"
                output = self.model.process(input, self.labels)
                return [Tensor("result", output)]
 ```

 In a scenario above the node runs some processing on the image with provided set of labels.
 When configuration allows for sending incomplete inputs, then the client can send labels only one time and then send only images.

 **Note**: Keep in mind that members of `OvmsPythonModel` objects are shared between **all** graph instances. It means that if in above scenario one client in one graph changes `labels`, then that change is also effective in every other graph instance (for every other client that sends requests to that graph). Saving data between executions that will be exclusive to a single graph instance is planned to be supported in future versions.

 ### Incomplete outputs

 `PythonExecutorCalculator` allows you to return incomplete set of outputs in `execute` method. It can be useful especially when working with streaming endpoints that serialize each graph output in a separate response. See an example:

 ```pbtxt
node {
  name: "python_node"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:input"
  output_stream: "OUTPUT:result"
  output_stream: "ERROR:error_message"
  node_options: {
    [type.googleapis.com / mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/model.py"
    }
  }
}
```

Python code that would run such node could look like this:

```python
from pyovms import Tensor

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        self.model = load_model(...)

    def execute(self, inputs: list):
        input = inputs[0]
        try:
            output = self.model(input)
        except Exception:
          return [Tensor("error_message", "Error occurred during execution".encode())]
        return [Tensor("result", output)]
 ```

In such case, the client could implement different actions depending on which output it receives on the stream.

Another example of such configuration is signaling that generation is finished when running in [generative mode](#generative-mode). This solution is used in [image generation demo](https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/stable_diffusion).


### Calculator type conversions

Python nodes work with a dedicated [Python Tensor](#python-tensor) objects that can be used both on C++ and Python side. The downside of that approach is that usually other calculators cannot read and create such objects. It means that Python nodes cannot be directly connected to any other, non-Python nodes.

That's why converter calculators exists. They work as adapters between nodes and implement necessary conversions needed to create a connection between calculators that work on two different types of packets.

#### PyTensorOvTensorConverterCalculator

OpenVINO Model Server comes with a built-in `PyTensorOvTensorConverterCalculator` that provides conversion between [Python Tensor](#python-tensor) and [OV Tensor](https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1_tensor.html).

Currently `PyTensorOvTensorConverterCalculator` works with only one input and one output.
- The stream that expects Python Tensor **must** have tag `OVMS_PY_TENSOR`
- The stream that expects OV Tensor **must** have tag `OVTENSOR`

In future versions converter calculator will accept multiple inputs and produce multiple outputs, but for now the only correct configuration is with one input stream and one output stream. One of those stream **must** have tag `OVMS_PY_TENSOR` and the other `OVTENSOR`, depending on the conversion direction.

`PyTensorOvTensorConverterCalculator` can also be configured to use node options with `tag_to_output_tensor_names` map and it's used in OV Tensor to Python Tensor conversion. It defines the name Python Tensor should be created with, based on output stream tag.

See a simplified example with both conversions taking place in the graph:

```pbtxt
input_stream: "OVMS_PY_TENSOR:input"
output_stream: "OVMS_PY_TENSOR:output"

node {
  name: "PythonPreprocess"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:input"
  output_stream: "OUTPUT:preprocessed_py"
  node_options: {
    [type.googleapis.com/mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/preprocess.py"
    }
  }
}

node {
  calculator: "PyTensorOvTensorConverterCalculator"
  input_stream: "OVMS_PY_TENSOR:preprocessed_py"
  output_stream: "OVTENSOR:preprocessed_ov"
}

node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session" # inference session
  input_stream: "OVTENSOR:preprocessed_ov"
  output_stream: "OVTENSOR:result_ov"
}

node {
  calculator: "PyTensorOvTensorConverterCalculator"
  input_stream: "OVTENSOR:result_ov"
  output_stream: "OVMS_PY_TENSOR:result_py"
  node_options: {
    [type.googleapis.com/mediapipe.PyTensorOvTensorConverterCalculatorOptions]: {
      tag_to_output_tensor_names {
        key: "OVMS_PY_TENSOR"
        value: "result_py"
      }
    }
  }
}

node {
  name: "PythonPostprocess"
  calculator: "PythonExecutorCalculator"
  input_side_packet: "PYTHON_NODE_RESOURCES:py"
  input_stream: "INPUT:result_py"
  output_stream: "OUTPUT:output"
  node_options: {
    [type.googleapis.com/mediapipe.PythonExecutorCalculatorOptions]: {
      handler_path: "/ovms/workspace/postprocess.py"
    }
  }
}
```

See a [CLIP demo](https://github.com/openvinotoolkit/model_server/tree/main/demos/python_demos/clip_image_classification) for a complete example of a graph that uses Python nodes, OV Inference nodes and converter nodes.
