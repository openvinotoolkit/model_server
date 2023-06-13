# Directed Acyclic Graph (DAG) Scheduler {#ovms_docs_dag}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_demultiplexing
   ovms_docs_custom_node_development


@endsphinxdirective

## Introduction
The Directed Acyclic Graph (DAG) Scheduler makes it possible to create a pipeline of models for execution in a single client request. 
The pipeline is a Directed Acyclic Graph with different nodes which define how to process each step of predict request. 
By using a pipeline, there is no need to return intermediate results of every model to the client. This allows avoiding the network overhead by minimizing the number of requests sent to the Model Server. 
Each model output can be mapped to another model input. Since intermediate results are kept in the server's RAM these can be reused by subsequent inferences which reduce overall latency.

This guide gives information about:

* <a href="#node-type">Node Types</a>
* <a href="#configuration-file">Configuration file</a>
* <a href="#using-pipelines">Using the pipelines</a>
* <a href="#pipeline-examples">Pipelines examples </a>
* <a href="#current-limitations">Current Limitations</a>



## Node Types <a name="node-type"></a>
### Auxiliary Node Types
There are two special kinds of nodes - Request and Response node. Both of them are predefined and included in every pipeline definition you create:
*  Request node
    - This node defines which inputs are required to be sent via gRPC/REST request for pipeline usage. You can refer to it by node name: `request`.
* Response node
    - This node defines which outputs will be fetched from the final pipeline state and packed into gRPC/REST response. 
    You cannot refer to it in your pipeline configuration since it is the pipeline final stage. To define final outputs fill `outputs` field. 

### Deep Learning node type

* DL model - this node contains underlying OpenVINO&trade; model and performs inference on the selected target device. This can be defined in the configuration file. 
    Each model input needs to be mapped to some node's `data_item` - input from gRPC/REST `request` or another `DL model` output. 
    Outputs of the node may be mapped to another node's inputs or the `response` node, meaning it will be exposed in gRPC/REST response. 

### Custom node type

* custom - that node can be used to implement all operations on the data which can not be handled by the neural network model. It is represented by
a C++ dynamic library implementing OVMS API defined in [custom_node_interface.h](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/src/custom_node_interface.h). Custom nodes can run the data
processing using OpenCV, which is included in OVMS, or include other third-party components. Custom node libraries are loaded into OVMS
 by adding their definition to the pipeline configuration. The configuration includes a path to the compiled binary with the `.so` extension. 
Custom nodes are not versioned, meaning one custom node library is bound to one name. To load another version, another name needs to be used.

    OpenVINO Model Server docker image comes with prebuilt custom nodes that you can use out-of-the-box in your pipeline. See the list of built-in custom nodes and
    learn more about developing custom nodes yourself in the [custom node developer guide](custom_node_development.md).

## Demultiplexing data

During the pipeline execution, it is possible to split a request with multiple batches into a set of branches with a single batch.
That way a model configured with a batch size 1, can process requests with arbitrary batch size. Internally, OVMS demultiplexer will
divide the data, process them in parallel and combine the results. 

De-multiplication of the node output is enabled in the configuration file by adding `demultiply_count`. 
It assumes the batches are combined on the first dimension which is dropped after splitting. For example:
- a node returns output with shape `[8,1,3,224,224]`
- demultiplexer creates 8 requests with shape `[1,3,224,224]`
- next model processes in parallel 8 requests with output shape `[1,1001]` each.
- results are combined into a single output with shape `[8,1,1001]`

[Learn more about demultiplexing](demultiplexing.md) 

## Configuration file <a name="configuration-file"></a>

Pipelines configuration is to be placed in the same json file like the 
[models config file](starting_server.md).
While models are defined in section `model_config_list`, pipelines are configured in
the `pipeline_config_list` section. 
Nodes in the pipelines can reference only the models configured in model_config_list section.

Basic pipeline section template is depicted below:

```

{
    "model_config_list": [...],
    "custom_node_library_config_list": [
        {
            "name": "custom_node_lib",
            "base_path": "/libs/libcustom_node.so"
        }
    ],
    "pipeline_config_list": [
        {
            "name": "<pipeline name>",
            "inputs": ["<input1>",...],
            "nodes": [
                {
                    "name": "<node name>",
                    "model_name": <reference to the model>,
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",  # reference to pipeline input
                                   "data_item": "<input1>"}}  # input name from the request
                    ], 
                    "outputs": [  # mapping the model output name to node output name
                        {"data_item": "<model output>",
                         "alias": "<node output name>"}
                    ] 
                },
                {
                    "name": "custom_node_name",
                    "library_name": "custom_node_lib",
                    "type": "custom",
                    "params": {
                        "param1": "value1",
                        "param2": "value2",
                    },
                    "inputs": [
                        {"input": {"node_name": "request",  # reference to pipeline input
                                   "data_item": "<input1>"}}  # input name from the request
                    ], 
                    "outputs": [
                        {"data_item": "<library_output>",
                            "alias": "<node_output>"},
                    ]
                }
            ],
            "outputs": [      # pipeline outputs
                {"label": {"node_name": "<node to return results>",
                           "data_item": "<node output name to return results>"}}
            ]
        }
    ]
}
```



### Pipeline configuration options explained

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"name"`|string|Pipeline identifier related to name field specified in gRPC/REST request|Yes|
|`"inputs"`|array|Defines input names required to be present in gRPC/REST request|Yes|
|`"outputs"`|array|Defines outputs (data items) to be retrieved from intermediate results (nodes) after pipeline execution completed for final gRPC/REST response to the client|Yes|
|`"nodes"`|array|Declares nodes used in pipeline and its connections|Yes|

### Node Options

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"name"`|string|Node name so you can refer to it from other nodes|Yes|
|`"model_name"`|string|You can specify underlying model (needs to be defined in `model_config_list`), available only for `DL model` nodes|required for `DL model` nodes|
|`"version"`|integer|You can specify a model version for inference, available only for `DL model` nodes|No|
|`"type"`|string|Node kind, currently there are 2 types available: `DL model` and `custom` |Yes|
|`"demultiply_count"`|integer|Splits node outputs to desired chunks and branches pipeline execution|No|
|`"gather_from_node"`|string|Setups node to converge pipeline and collect results into one input before execution|No|
|`"inputs"`|array|Defines the list of input/output mappings between this and dependency nodes, **IMPORTANT**: Please note that output shape, precision, and layout of previous node/request needs to match input of current node's model|Yes|
|`"outputs"`|array|Defines model output name alias mapping - you can rename model output names for easier use in subsequent nodes|Yes|

### Node Input Options

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"node_name"`|string|Defines which node we refer to|Yes|
|`"data_item"`|string|Defines which resource of the node we point to|Yes|

### Node Output Options

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"data_item"`|string|Is the name of resource exposed by node - for `DL model` nodes it means model output|Yes|
|`"alias"`|string|Is a name assigned to a data item, makes it easier to refer to results of this node in subsequent nodes|Yes|


### Custom Node Options

In case the pipeline definition includes the custom node, the configuration file must include `custom_node_library_config_list`
section. It includes:
|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"name"`|string|The name of the custom node library - it will be used as a reference in the custom node pipeline definition |Yes|
|`"base_path"`|string|Path the dynamic library with the custom node implementation|Yes|

Custom node definition in a pipeline configuration is similar to a model node. Node inputs and outputs are configurable in 
the same way. Custom node functions are just like a standard node in that respect. The differences are in the extra parameters:

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"library_name"`|string|Name of the custom node library defined in `custom_node_library_config_list`|Yes|
|`"type"`|string|Must be set to `custom`|Yes|
|`"params"`| json object with string values| a list of parameters and their values which could be used in the custom node implementation|No|

## Using Pipelines <a name="using-pipelines"></a>

Pipelines can use the same API as the models. There are exactly the same calls for running 
the predictions. The request format must match the pipeline definition inputs.


The pipeline configuration can be queried using [gRPC GetModelMetadata](model_server_grpc_api_tfs.md) calls and
[REST Metadata](model_server_rest_api_tfs.md).
It returns the definition of the pipelines inputs and outputs. 

Similarly, pipelines can be queried for their state using the calls [GetModelStatus](model_server_grpc_api_tfs.md)
and [REST Model Status](model_server_rest_api_tfs.md)

The only difference in using the pipelines and individual models is in version management. In all calls to the pipelines, 
the version parameter is ignored. Pipelines are not versioned. Though, they can reference a particular version of the models in the graph.

## Pipelines Examples <a name="pipeline-examples"></a>

[Single face analysis with combined models](../demos/single_face_analysis_pipeline/python/README.md)

[Multiple vehicles analysis using demultiplexer with model_zoo_object_detection example custom node](../demos/vehicle_analysis_pipeline/python/README.md)

[Optical Character Recognition pipeline with east_ocr example custom node](../demos/optical_character_recognition/python/README.md)

[Horizontal Text Detection pipeline with horizontal_ocr example custom node](../demos/horizontal_text_detection/python/README.md)

## Current limitations <a name="current-limitations"></a>

- Models with "auto" [batch size](dynamic_bs_auto_reload.md) or [shape](dynamic_shape_auto_reload.md) cannot be referenced in pipeline
- Connected inputs and output for subsequent node models need to match each other in terms of data shape, precision and layout - 
there is no automatic conversion between input/output model precisions or layouts. This limitation can be addressed with `--shape` and `--layout` model configuration or with a custom node to transform the data as required to match the expected data format.
- REST requests with no named format (JSON body with one unnamed input) are not supported


## See Also

- [Optimization of Performance](./performance_tuning.md)
