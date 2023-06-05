# Custom Node Development Guide {#ovms_docs_custom_node_development}

## Overview

Custom node in OpenVINO Model Server simplifies linking deep learning models into complete pipelines even if the inputs and output
of the sequential models do not fit. In many cases, the output of one model can not be directly passed to another one.
The data might need to be analyzed, filtered, or converted to a different format. The operations can not be easily implemented
in AI frameworks or are simply not supported. Custom node addresses this challenge. They allow employing a dynamic library
developed in C++ or C to perform arbitrary data transformations. 

## Custom Node API


The custom node library must implement the API interface defined in [custom_node_interface.h](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_node_interface.h).
The interface is defined in `C` to simplify compatibility with various compilers. The library could use third party components
linked statically or dynamically. OpenCV is a built in component in OVMS which could be used to perform manipulation on the image
data. 

The data structure and functions defined in the API header are explained below. 

### "CustomNodeTensor" struct 

The CustomNodeTensor struct consist of several fields defining the data in the output and input of the node execution.
Custom node can generate results based on multiple inputs from one or more other nodes. 
"Execute" function has access to the pointer to multiple CustomNodeTensor objects which stores multiple inputs to be processed.
Each input can be referenced using an index or you can search by name:
```
inputTensor0 = &(inputs[0])
```
Every CustomNodeTensor struct includes the following fields:
- `const char* name`  - pointer to the string representing the input name.
- `uint8_t* data` - pointer to data buffer. Data is stored as bytes.
- `uint64_t dataBytes` - the size of the data allocation in bytes.
- `uint64_t* dims` - pointer to the buffer storing array shape size. Size of each dimension consumes 8 bytes.
- `uint64_t dimsCount` - number of dimensions in the data array.
- `CustomNodeTensorPrecision precision` - data precision enumeration.

### CustomNodeTensorInfo struct

The fields in struct CustomNodeTensorInfo are similar to CustomNodeTensor. It holds information about 
the metadata of the custom node interfaces: inputs and outputs.

### "CustomNodeParam" struct

Struct CustomNodeParam stores a list of pairs with pointers to the parameter and value strings.
Each parameter in such objects can be referenced using an index which you can search by key name by iterating the structure.

### "execute" function
```
int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount);
```

This function implements the data transformation of the custom node. The input data for the function are passed in the form of 
a pointer to CustomNodeTensor struct object. It includes all the data and pointers to buffers for all custom node inputs.
The parameter inputsCount pass info about the number of inputs passed that way.

Note that the execute function should not modify the buffers storing the input data because that would alter the data
which potentially might be used in other pipeline nodes.

The behavior of the custom node execute function can depend on the node parameters set in the OVMS configuration.
They are passed to the execute function in `params` argument. `paramsCount` passes the info about the number of parameters configured.

The results of the data transformation should be returned by the outputs pointer to a pointer that stores the address of 
CustomNodeTensor struct. The number of outputs is defined during the function execution in the `outputsCount` argument.

Note that during the function execution all the output data buffers need to be allocated. They will be released by OVMS after 
the request processing is completed and returned to the user. The cleanup is triggered by calling the `release` function 
which also needs to be implemented in the custom library.

In some cases, dynamic allocation in `execute` call might be a performance bottleneck or cause memory fragmentation. Starting from 2022.1 release, it is possible to preallocate memory during DAG initialization and reuse it in subsequent inference requests. Refer to `initialize` and `deinitialize` functions below. Those can be used to implement preallocated memory pool. Example implementation can be seen in [custom node example source](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/src/custom_nodes/add_one/add_one.cpp#L141).

Execute function returns an integer value that defines the success (`0` value) or failure (other than 0). When the function 
reports error, the pipeline execution is stopped and the error is returned to the user. 

### "getInputsInfo" function
This function returns information about the metadata of the expected inputs. Returned CustomNodeTensorInfo object is used 
to create a response for getModelMetadata calls. It is also used in the user request validation and pipeline 
configuration validation.

Custom nodes can generate the results which have dynamic size depending on the input data and the custom node parameters.
In such case, function `getInputsInfo` should return value `0` on the dimension with dynamic size. It could be input with
variable resolution or batch size. 

### "getOutputInfo" function
Similar to the previous function but defining the metadata of the output.

### "release" function
This function is called by OVMS at the end of the pipeline processing. It clears all memory allocations used during the 
node execution. This function should call `free` if `malloc` was used to allocate output memory in `execute` function. The function should return preallocated memory to the pool if memory pool was used. OVMS decides when to free and which buffer to free.

### "initialize" function
```
int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount);
```
This function enables creation of resources to be reused between predictions. Potential use cases include optimized temporary buffers allocation. Using `initialize` is optional and not required for custom node to work. `customNodeLibraryInternalManager` should be instantiated inside this function if initialize is used. On initialize failure, status not equal to `0` should be returned to make OVMS treat it as an error.

When not used, minimal dummy implementation is required. Return `0`, meaning no error:
```
int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}
```

### "deinitialize" function
```
int deinitialize(void* customNodeLibraryInternalManager);
```
This function enables destruction of resources that were used between predictions. Using deinitialize is optional and not required for custom node to work. `customNodeLibraryInternalManager` should be destroyed here if deinitialize is used. On deinitialization failure, status not equal to `0` should be returned to make OVMS treat it as an error.

When not used, minimal dummy implementation is required. Return `0`, meaning no error:
```
int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}
```

## Using OpenCV
The custom node library can use any third-party dependencies which could be linked statically or dynamically.
For simplicity OpenCV libraries included in the OVMS docker image can be used.
Just add include statement like:

```
#include "opencv2/core.hpp"
```

## String support
There are special consideration when handling in the custom nodes the input sent by the clients as string. Such data when received by the OVMS frontend, is automatically converted to a 2D array with shape [-1,-1]. Example of custom node using this feature is our [Tokenizer](https://github.com/openvinotoolkit/model_server/tree/develop/src/custom_nodes/tokenizer). 

### inputs
When strings are send to the custom node that has 2-dimensional shape and U8 precision OVMS, after receiving request containing such inputs converts them to the 2 dimensional U8 array of  shape [number of strings, length of the longest string + 1] with padding filled with zeros. For example batch of three strings ["String_123", "", "zebra"] would be converted to:
```
['S', 't', 'r', 'i', 'n', 'g', '_', '1', '2', '3', 0,  // String_123
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                       // ""
'z', 'e', 'b', 'r', 'a', 0, 0, 0, 0, 0, 0]             // "zebra"
```

### outputs

When the name of the custom node output is suffixed with _string, its shape has 2 dimensions and precision is U8 OVMS treats data of such output as array that contains string in every row with padding filled with zeros and convert automatically data of such outputs to strings. For example U8 array:
```
['S', 't', 'r', 'i', 'n', 'g', '_', '1', '2', '3', 0,  // String_123
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                       // ""
'z', 'e', 'b', 'r', 'a', 0, 0, 0, 0, 0, 0]             // "zebra"
```
would be converted to ["String_123", "", "zebra"].

## Building

Custom node library can be compiled using any tool. It is recommended to follow the example based 
a docker container with all build dependencies included. It is described in this [Makefile](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/east_ocr/Makefile). 

## Testing 
The recommended method for testing the custom library is via OVMS execution:
- Compile the library using a docker container configured in the Makefile. It will be exported to `lib` folder.
- Prepare a pipeline configuration with the path custom node library compiled in the previous step.
- Start OVMS docker container.
- Submit a request to OVMS endpoint using a gRPC or REST client.
- Analyse the logs on the OVMS server.

For debugging steps, refer to the OVMS [developer guide](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/docs/developer_guide.md)


## Built-in custom nodes
Along with the OpenVINO Model Server, there are also built-in custom nodes provided in the image.
They reside in the `/ovms/lib/custom_nodes` directory in the container and can be referred in the configuration file.
Below you can see the list of fully functional custom nodes embedded in the model server docker image:

| Custom Node      | Location in the container |
| :---        |    :----   |
| [east-resnet50 OCR custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/east_ocr) | `/ovms/lib/custom_nodes/libcustom_node_east_ocr.so`|
| [horizontal OCR custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/horizontal_ocr) | `/ovms/lib/custom_nodes/libcustom_node_horizontal_ocr.so`|
| [model zoo intel object detection custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/model_zoo_intel_object_detection) | `/ovms/lib/custom_nodes/libcustom_node_model_zoo_intel_object_detection.so`|
| [image transformation custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/image_transformation) | `/ovms/lib/custom_nodes/libcustom_node_image_transformation.so`|
| [add one custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/add_one) | `/ovms/lib/custom_nodes/libcustom_node_add_one.so`|
| [face blur custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/face_blur) | `/ovms/lib/custom_nodes/libcustom_node_face_blur.so`|


**Example:** 
Including built-in [horizontal OCR custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/horizontal_ocr) in the `config.json` would look like:
```json
...
    "custom_node_library_config_list": [
        {
            "name": "ocr_image_extractor",
            "base_path": "/ovms/lib/custom_nodes/libcustom_node_horizontal_ocr.so"
        }
    ],
...
```

The custom node is already available under this path. No need to build anything and mounting to the container. 

Additional examples are included in the unit tests:
- [node_add_sub.c](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/test/custom_nodes/node_add_sub.c)
- [node_choose_maximum.cpp](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/test/custom_nodes/node_choose_maximum.cpp)
- [node_missing_implementation.c](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/test/custom_nodes/node_missing_implementation.c)
- [node_perform_different_operations.cpp](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/test/custom_nodes/node_perform_different_operations.cpp)

