# Custom Node Development Guide

## Overview

Custom node in OpenVINO Model Server simplifies linking deep learning models into a complete pipelines even if the inputs and output
of the sequential models do not fit. In many cases, output of one model can not be directly passed to another one.
The data might need to be analysed, filtered or converted to difference format. Those operations can not be easily implemented
in AI frameworks or are just not supported. Custom node address this challenge. They allows employing a dynamic library
developed in C++ or C to perform arbitrary data transformations. 

## Custom Node API

The custom node library must implemented the API interface defined in [custom_node_interface.h](../src/custom_node_interface.h).
The interface is defined in `C` to simplify compatibility with various compilers. The library could use third party components
linked statically or dynamically. OpenCV is a built in component in OVMS which could be used to perform manipulation on the image
data. 

Below are explained the data structures and functions defined in the API header. 

### "CustomNodeTensor" struct 

The CustomNodeTensor struct consist of several fields defining the data in the output and input of the node execution.
Because it has constant length, it can carry multiple inputs. Custom node can generate results based on multiple inputs 
from one or more other nodes. Each input can be referenced using an index:
```
inputTensor0 = &(inputs[0])
```
Every CustomNodeTensor struct inlcude the following fields:
`const char* name`  - pointer to the string representing the input name
`uint8_t* data` - pointer to data buffer. Data is stored as bytes.
`uint64_t dataBytes` - the size of the data allocation in bytes
`uint64_t* dims` - pointer to the buffer storing array shape size. Size of each dimension consumes 8bytes
`uint64_t dimsCount` - number of dimension on the data array
`CustomNodeTensorPrecision precision` - data precision selector

### CustomNodeTensorInfo struct

The fields in struct CustomNodeTensorInfo are similar like in CustomNodeTensor. It just holds information about 
the metadata of the custom node interfaces: inputs and outputs.

### "CustomNodeParam" struct

Struct CustomNodeParam stores a list of pairs with pointers to parameter and and value strings.
Each parameter can be references in such object using an index.

### "execute" function
```
int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount);
```

This function implements the data transformation of the custom node. The input data for the function are passed in the form of 
a pointer to CustomNodeTensor struct object. It includes all the data and pointers to buffers for all custom node inputs.
The parameter inputsCount pass info about the number of inputs passed that way.

Note that the execute function should not modify the buffers storing the input data because that would alter the data
which potentially might be used in other pipeline nodes.

The behaviour of custom node execute function can be dependent on the node parameters set in OVMS configuration.
They are passed to execute function in `params` argument. paramsCount pass info about the number of parameters configured.

The results of the data transformation should be returned by the outputs pointer to a pointer which stores the address of 
CustomNodeTensor struct. Number of outputs is to be defined during the function execution in the `outputsCount` argument.

Note that during the function execution all the output data buffers needs to be allocated. They will be released by OVMS after 
the request processing is completed and returned to the user. The cleanup is triggered by calling `release` function.

Execute function returns an integer which value defines the success (0) or failure (1). When the function 
reports error, the pipeline execution is stopped and error is returned to the user.
A special response is returned (2) when the execute doesn't return any results. It could be the case when object detection 
model doesn't detect any valid objects in the input image.

### "getInputsInfo" function
This function returns information about the metadata of the expected inputs. Returned CustomNodeTensorInfo object is used 
to create a response for getModelMetadata calls. It is also used in the user request validation and the pipeline 
configuration validation.

Custom nodes can generate the results which have dynamic size depending on the input data and the custom node parameters.
In such case function getInputsInfo should return value `-1` on the dimension with dynamic size. It could be an input with
variable resolution or a batch size.

### "getOutputInfo" function
Similar to previous function but defining the outputs metadata.

### "release" function
This function is called by OVMS at the end of the pipeline processing. It should clear all memory allocations used during the 
node execution.

## Using OpenCV
The custom node library can use any third-party dependencies which could be linked statically or dynamically.
For simplicity there could be used OpenCV libraries which are already included in the OVMS docker image.
Just add include statement like:
```c++
#include "opencv2/core.hpp"
```

## Building
Custom node library can be compiled using any tool but for simplicity there are prepared a method based on a g++ compiled 
inside a docker container.
Also bazel could be adopted like in the example


## Testing 
Recommended method for tesing the custom library is via OVMS execution:
- Compile the library using a docker container like configured in the Makefile. It will be exported to `lib` folder.
- Prepare a pipeline configuration with the path custom node library compiled in the previous step
- Start OVMS docker container
- Submit a request to OVMS endpoint using a gRPC or REST client
- Analyse the logs on the OVMS server

For debugging steps refer the OVMS [developer guide](developer_guide.md)


## Custom node examples 
The best staring point for developing new custom nodes is via exploring or copying the examples.

The fully functional custom nodes are in:
- [east-resnet50 OCR custom node](../src/custom_nodes/east_ocr)

Other examples are included in the unit tests:
- [node_add.c](../src/test/custom_nodes/node_add_sub.c)
- [node_choose_maximum.cpp](../src/test/custom_nodes/node_choose_maximum.cpp)
- [node_add.c](../src/test/custom_nodes/node_missing_implementation.c)
- [node_mock.c](../src/test/custom_nodes/node_mock.c)
- [node_perform_different_operations.cpp](../src/test/custom_nodes/node_perform_different_operations.cpp)


