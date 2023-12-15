# Python support in OpenVINO Model Server - preview {#ovms_docs_python_support_python_support}


5. PythonExecutorCalculator And Mediapipe Graphs
6. Usage With gRPC Streaming
> timestamping and output gathering
> single request multiple response (cycles)

## Introduction

**This feature is in preview, meaning some behviors of the feature as well as user interface are subjects to change in the future versions**

 Starting with version 2023.3, model server introduced support for execution of custom Python code. Users can now create their own servables in Python. Those servables can run simple pre or post processing tasks as well as complex ones like image or text generation. 
 
 Python is enabled via [Mediapipe](../mediapipe.md) by built-in `PythonExecutorCalculator` that allows creating graph nodes that execute Python code. That way Python servables can be used as standalone endpoints (single node graphs) or be part of larger Mediapipe solutions.

 Checkout [quickstart guide](quickstart.md) for simple usage example.

*Note:* This feature is currently in preview.

 ## Building Docker Image

Publically distributed Docker images support Python, but they do not come with any external modules. If bare Python is all you need then you can use public image directly. Otherwise you need to extend public image with additional layers that will install all the stuff you need for your Python code to work. For example, let's say you need numpy.
In that case your Dockerfile may look like:

```dockerfile
FROM openvino/model_server:latest
USER root
ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python
RUN apt update && apt install -y python3-pip git
RUN pip3 install numpy
ENTRYPOINT [ `/ovms/bin/ovms` ]
```

TODO: Here we could provide make target with requirements location as a parameter.

### Building Model Server From Source
In above section you use public Docker image. In case you want to build it from source do the following:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make docker_build MEDIAPIPE_DISABLE=0 PYTHON_DISABLE=0 OV_USE_BINARY=1 RUN_TESTS=0
cd ..
```
Resulting Docker image can be extended with additional layers just as the public one.

## Python Servable

When deploying Python servable, Model Server expects Python file with an `OvmsPythonModel` class implemented:

```python
class OvmsPythonModel:

    def initialize(self, kwargs):
        """
        `initialize` is called when model server loads graph definition. It allows to initialize and maintain state between subsequent execute() calls and even graph instances. 
        For gRPC unary, graphs are recreated per request. 
        For gRPC streaming, there can be multiple graph instances existing at the same time. 
        OvmsPythonModel object is initialized with this method and then shared between all graph instances. Implementing this function is required (should it be?).

        Parameters:
        -----------
        kwargs : dict
            Available arguments:
            * node_name: string
                Name of the node in the graph
            * input_streams: list of strings
                List of input stream names defined for the node in graph configuration
            * output_streams: list of strings
                List of output stream names defined for the node in graph configuration
        -----------
        """
        print("Running initialize...")

    def execute(self, inputs):
        """
        `execute` is called in `Process` method of PythonExecutorCalculator which in turn is called by Mediapipe framework. How Mediapipe calls `Process` method for the node depends on the configuration and the two configurations supported by PythonExecutorCalculator are:
        
        * Regular: `execute` runs every time the node receives inputs. Produces one set of outputs per one set of inputs. For unary endpoints it's the only possible configuration.

        * Generative: `execute` runs multiple times for single inputs set. Produces multiple sets of outputs over time per single set of inputs. Works only with streaming endpoints. 

        Implemeting this function is required.

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
        `finalize` is called when model server unloads graph definition. It allows to perform any cleanup actions before the graph is removed. Implementing this function is optional.
        """
        print("Running finalize...")
```

### initialize

`initialize` is called when model server loads graph definition. It allows to initialize and maintain state between subsequent `execute` calls and even graph instances.

For gRPC unary, graphs are recreated per request.

For gRPC streaming, there can be multiple graph instances existing at the same time.

`OvmsPythonModel` object is initialized with this method and then shared between all graph instances.

#### Parameters and return value

`initialize` is called with `kwargs` parameter which is a dictionary. All keys are strings. Available keys and values:

| Key           | Value type | Description |
| ------------- |:-----------| :-----------| 
| node_name     | string | Name of the node in the graph |
| input_names   | list of strings | List of input stream names defined for the node in graph configuration | 
| outputs_names | list of strings | List of output stream names defined for the node in graph configuration |

`initialize` is not expected to return any value.

#### Error handling

Signaling that something went wrong should be done by throwing an exception.
When model server catches exception from `initialize` it cleans up all Python resources in the graph (including those belonging to the correctly loaded nodes) and sets the whole graph in invalid state.

**Implementing this function is required (should it be?).**

### execute

`execute` is called in `Process` method of `PythonExecutorCalculator` which in turn is called by Mediapipe framework. How Mediapipe calls `Process` for the node depends on the configuration and the two configurations supported by `PythonExecutorCalculator` are:

#### Regular 

`execute` runs every time the node receives complete set inputs with the same timestamp. It produces one set of outputs per one set of inputs. For unary endpoints it's the only possible configuration. On the implementation side, to use that mode, `execute` should `return` outputs.

```python
def execute(self, inputs):
    ...
    return outputs
```

More information along with the configuration aspect described can be found in [execution modes]() section.

#### Generative 

`execute` runs multiple times for single set of inputs with the same timestamp. It produces multiple sets of outputs over time per single set of inputs. Works only with streaming endpoints. On the implementation side, to use that mode, `execute` should `yield` outputs.

```python
def execute(self, inputs):
    # For single set on inputs generate 10 sets of outputs
    for _ in range(10):
        ... 
        yield outputs
```

More information along with the configuration aspect described can be found in [execution modes]() section.

#### Parameters and return value

`execute` is called with `inputs` parameter which is a `list of pyovms.Tensor`. 

Depending on the mode it should return:

- For regular mode: `list of pyovms.Tensor`
- For generative mode: `generator` that generates `list of pyovms.Tensor`

So depending on the mode `execute` must always either `return` or `yield` a `list of pyovms.Tensor`

*Note*: This method returns outputs as a full set, but since each output is a separate packet in Mediapipe flow, they do not arrive together to their destination. Be aware that if you have more than one output and outputs of your Python node are also outputs of the whole graph you will receive each output in a separate response. You can then gather them using timestamp that can be found in the response. TODO: more about it...

#### Error handling

Signaling that something went wrong should be done by throwing an exception.
The exception is caught by the `PythonExecutorCalculator` which logs it and returns non-OK status.
Model Server then reads that status and sets graph in an error state. Then it closes all graph's input streams and waits until in-progress actions are finished. Once it's done the graph gets removed.

This behavior has different effect on the client depending on the kind of gRPC endpoint used - unary or streaming.

**Unary** 

With unary endpoint a graph is created, executed and destroyed for every request. If `execute` encounters an error, model server logs it and sends error message in response immediately. 

**Streaming**

With streaming endpoint a graph is created for the first request in the stream and then reused by all subsequent requests. 

If `execute` encounters an error on the first request (for example the Python code doesn't work as expected), model server logs it  and sends error message in response immediately. The graph gets destroyed.

If `execute` encounters an error on one of the subsequent requests (for example wrong data has been received), model server logs it and Mediapipe sets error in the graph, but the client won't receive error message until it sends another request. When the next request is read from the stream, model server checks if graph has an error, destroys it and sends response to the client.

As of now, the graphs are not recoverable so if an error occurs, you need to create a new stream.

**Implementing this function is required.**

### finalize

`finalize` is called when model server unloads graph definition. It allows to perform any cleanup actions before the graph is removed. 

#### Parameters and return value

`finalize` does not have any parameters and is not expected to return any value.

#### Error handling

Signaling that something went wrong should be done by throwing an exception.
When model server catches exception from `finalize` it logs it and proceeds with the unload.

*Implementing this function is optional.*


## Python Tensor

`PythonExecutorCalculator` operates on a dedicated `Tensor` class that wraps the data along with some additional information like name, shape or datatype. Objects of that class are passed to `execute` method as inputs and returned as output. They are also wrapped and exchanged between nodes in the graph and between graph and model server core. 

This `Tensor` class is a C++ class with a Python binding that implements Python [Buffer Protocol](https://docs.python.org/3/c-api/buffer.html#buffer-protocol). It can be found in a built-in module `pyovms`.

### Accessing Tensor Contents

`pyovms.Tensor` attributes:

| Name           | Type | Description |
| ------------- |:-----------| :-----------| 
| name     | string | Name of the string that also assiciates it with input or output stream of the node |
| shape   | tuple | Tuple of numbers defining the shape of the tensor | 
| datatype | string | Type of elements in the buffer. |
| data | memoryview | Memoryview of the underlaying data buffer |
| size | number | Size of data buffer in bytes |

*Note*: `datatype` attribute is not part of buffer protocol implementation.
Buffer protocol uses `format` value that uses [struct format characters](https://docs.python.org/3/library/struct.html#format-characters). It can be read from `data` memoryview. 
There's a mapping between those two - see [datatype considerations](...).

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

`Tensor(tensor_name, data)`

- `tensor_name`: a string that assosiates Tensor data with specific name. This name is also used by `PythonExecutorCalculator` to push data to the correct output stream in the node. More about it in [Mediapipe section](...).
- `data`: an object that implements Python [Buffer Protocol](https://docs.python.org/3/c-api/buffer.html#buffer-protocol). This could be an instance of some built-in type like `bytes` or types from external modules like `numpy.ndarray`. 

```python
from pyovms import Tensor

class OvmsPythonModel:
    def execute(self, inputs):
        # Create Tensor called my_output with encoded text
        output = Tensor("my_output", "some text".encode())
        # A list of Tensors is expected, even if there's only one ouput
        return [output]
``` 

As `Tensor` gets created from another type it adapts all fields required by the buffer protocol as its own. More about this in [datatype considerations](...) section.

If the node is connected to another Python node, then Tensors pushed to the output of this node, are inputs of another node. 

### Datatype considerations

There are two places where `pyovms.Tensor` objects are created and accessed:
- in `execute` method of `OvmsPythonModel` class
- in model server core during serialization and deserialization if Python node inputs or outputs as also graph inputs or outputs

Model Server receives requests and sends responses on gRPC inferface via KServe API which defines [expected data types for tensors](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data-types).
On the other hand Python [Buffer Protocol](https://docs.python.org/3/c-api/buffer.html#buffer-protocol) requires `format` to be specified as [struct format characters](https://docs.python.org/3/library/struct.html#format-characters). 

In order to let users work with KServe types without enforcing the usage of struct format characters on the client side, model server attempts to do the mapping as follows:

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


The same mapping is applied the other way around during response serialization.
In some cases, users may work with more complex types that are not listed above and model server also allows that.

#### Custom types

The `datatype` field in the tensor is a `string` and model server will not reject datatype that is not among above KServe types. If some custom type is defined in the request and server cannot map it to a format character it will translate it to `B` treating it as a 1-D raw binary buffer. For consistency the shape of the underlaying buffer in that case will also differ from the shape defined in the request. Let's see it on the example:

1. Model Server receives request with the follwing input:
    * datatype: "my_string"
    * shape: (3,)
    * data: "null0terminated0string0" string encoded in UTF-8

2. Model Server creates `pyovms.Tensor` with:
    * Tensor.datatype: "my_string"
    * Tensor.shape: (3,)
    * Tensor.data.format: "B"
    * Tensor.data.shape: (23,)
    
In `execute` user has access to both information from the request as well as how the internal buffer looks like.

Above scenario is the case only for the nodes that are directly connected to graph inputs. `pyovms.Tensor` objects produced inside `execute` inherit most of the fields from the objects they are created from, so user cannot manually set datatype. In such case tensor will try to map buffer protocol `format` to `datatype` according to the mapping mentioned before. 

If it fails, the `datatype` is set to `format`, so that if such tensor is the output tensor of the graph, client receives the most valuable information about the type of output data.
  
## Configuration and deployment

Python is enabled via [Mediapipe](../mediapipe.md) by built-in `PythonExecutorCalculator`, therefore, in order to execute Python code in OVMS you need to create a graph with a node that uses this calculator. 

The way the graph is configured has a huge impact on the whole deployment. It defines things like:
- inputs and outputs of the graph
- inputs and outputs of each node in the graph
- connections between the nodes
- graph and nodes options

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

- `input_stream`: defines input in form `[TAG]:[NAME]`. Mediapipe allows configurations with indexes i.e. `[TAG]:[INDEX]:[NAME]`, but `PythonExecutorCalculator` ignores it.

- `output_stream`: defines output in form `[TAG]:[NAME]`. Mediapipe allows configurations with indexes i.e. `[TAG]:[INDEX]:[NAME]`, but `PythonExecutorCalculator` ignores it.

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

### Input and output streams for entire graph

So far only node input and output streams have been mentioned, but the configuration also requires defining graph's input and output streams.
The rules are very similar to how it works on the node level, so the streams are described in form: `[TAG]:[NAME]`, but there's more to it.

On graph level the `[TAG]` helps model server in deserialization and serialization by providing information about the object type expected in the stream. Model server reads the tag and expects it to start with one of predefined prefixes. If graph input stream is connected to Python node the tag should begin with `OVMS_PY_TENSOR`, which tells the server that it should deserialize input from the request to `pyovms.Tensor` object.

There can't be two or more the same tags among the input streams as well as there can't be two or more the same tags among the output streams. In such cases, prefix must be followed by some unique string.

```
input_stream: "OVMS_PY_TENSOR_IMAGE:image"
input_stream: "OVMS_PY_TENSOR_TEXT:text"
output_stream: "OVMS_PY_TENSOR:output"
```

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

In that example client will send an input called `first_number` and receive output called `last_number`. Since user has access to input and output names in the Python code, the code for incrementation can be generic.

`incrementer.py`
```python
from pyovms import Tensor

def increment(input):
    # Some code for input incrementation
    ...

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
Where `name` defines the name of the whole servable and `graph_path` contains the path to graph configuration file.

## Advanced Configuration

### Execution modes

Python nodes can be configured to run in two execution modes - regular and generative. 

In regular execution mode the node produces one set of outputs per one set of inputs. It works via both gRPC unary and streaming endpoints and is a common mode for use cases like computer vision.

In generative execution mode the node produces multiple sets of outputs over time per single set of inputs. It works only via gRPC streaming endpoints and is useful for use cases where total processing time is big and you want to return some intermediate results before the exection is completed. That mode is well suited to Large Language Models to serve them in more interactive manner.

Depending on which mode is used, both the Python code and graph configuration must be in line.

