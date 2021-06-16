# Directed Acyclic Graph Scheduler in OpenVINO&trade; Model Server

## Introduction
OpenVINO&trade; Model Server provides possibility to create pipeline of models for execution in a single client request. 
Pipeline is a Directed Acyclic Graph with different nodes which define how to process each step of predict request. 
By using such pipeline, there is no need to return intermediate results of every model to the client. This allows avoiding the network overhead by minimizing the number of requests sent to model server. 
Each model output can be mapped to another model input. Since intermediate results are kept in server's RAM these can be reused by subsequent inferences which reduces overall latency.

This guide gives information about following:

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
    - This node defines which outputs will be fetched from final pipeline state and packed into gRPC/REST response. 
    You cannot refer to it in your pipeline configuration since it is the pipeline final stage. To define final outputs fill `outputs` field. 

### Deep Learning node type

* DL model - this node contains underlying OpenVINO&trade; model and performs inference on selected target device. This can be defined in configuration file. 
    Each model input needs to be mapped to some node's `data_item` - input from gRPC/REST `request` or another `DL model` output. 
    Outputs of the node may be mapped to another node's inputs or the `response` node, meaning it will be exposed in gRPC/REST response. 

### Custom node type

* custom - that node can be used to implement all operations on the data which can not be handled by the neural network model. It is represented by
a C++ dynamic library implementing OVMS API defined in [custom_node_interface.h](../src/custom_node_interface.h). Custom nodes can run the data
processing using OpenCV, which is included in OVMS, or include other third-party components. Custom node libraries are loaded into OVMS
 by adding its definition to the pipeline configuration. The configuration includes a path to the compiled binary with `.so` extension. 
Custom nodes are not versioned, meaning one custom node library is bound to one name. To load another version, another name needs to be used.

Learn more about developing custom node in the [custom node developer guide](custom_node_development.md)

## Demultiplexing data

During the pipeline execution, it is possible to split a request with mulitple batches into a set of branches with a single batch.
That way a model configured with a batch size 1, can process requests with arbitrary batch size. Internally, OVMS demultiplexer will
divide the data, process them in parallel and combine the results. 

De-multiplication of the node output is enabled in the configuration file by adding `demultiply_count`. 
It assumes the batches are combined on the first dimension which is dropped after splitting. For example:
- a node returns output with shape `[8,1,3,224,224]`
- demuliplexer creates 8 requests with shape `[1,3,224,224]`
- next model processes in parallel 8 requests with output shape `[1,1001]` each.
- results are combined into a single output with shape `[8,1,1001]`

[Learn more about demuliplexing](demultiplexing.md) 

## Configuration file <a name="configuration-file"></a>

Pipelines configuration is to be placed in the same json file like the 
[models config file](docker_container.md#configfile).
While models are defined in section `model_config_list`, pipelines are to be configured in
section `pipeline_config_list`. 
Nodes in the pipelines can reference only the models configured in model_config_list section.

Below is depicted a basic pipeline section template:

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
                    "name": "custon_node_name",
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
|`"name"`|string|Pipeline identifier related to name field specified in gRPC/REST request|&check;|
|`"inputs"`|array|Defines input names required to be present in gRPC/REST request|&check;|
|`"outputs"`|array|Defines outputs (data items) to be retrieved from intermediate results (nodes) after pipeline execution completed for final gRPC/REST response to the client|&check;|
|`"nodes"`|array|Declares nodes used in pipeline and its connections|&check;|

### Node options explained

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"name"`|string|Node name so you can refer to it from other nodes|&check;|
|`"model_name"`|string|You can specify underlying model (needs to be defined in `model_config_list`), available only for `DL model` nodes|required for `DL model` nodes|
|`"version"`|integer|You can specify model version for inference, available only for `DL model` nodes||
|`"type"`|string|Node kind, currently there is only `DL model` kind available|&check;|
|`"demultiply_count"`|integer|Splits node outputs to desired chunks and branches pipeline execution||
|`"gather_from_node"`|string|Setups node to converge pipeline and collect results into one input before execution||
|`"inputs"`|array|Defines list of input/output mappings between this and dependency nodes, **IMPORTANT**: Please note that output shape, precision and layout of previous node/request needs to match input of current node's model|&check;|
|`"outputs"`|array|Defines model output name alias mapping - you can rename model output names for easier use in subsequent nodes|&check;|

### Node input options explained

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"node_name"`|string|Defines which node we refer to|&check;|
|`"data_item"`|string|Defines which resource of node we point to|&check;|

### Node output options explained

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"data_item"`|string|Is the name of resource exposed by node - for `DL model` nodes it means model output|&check;|
|`"alias"`|string|Is a name assigned to data item, makes it easier to refer to results of this node in subsequent nodes|&check;|


### Custom node options explained

In case the pipeline definition includes the custom node, the configuration file must include `custom_node_library_config_list`
section. It includes:
|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"name"`|string|The name of the custom node library - it will be used as a reference in the custom node pipeline definition |&check;|
|`"base_path"`|string|Path the the dynamic library with the custom node implementation|&check;|

Custom node definition in the pipeline configuration is similar to the model node. Node inputs and outputs are configurable in 
the same way. Custom node functions just like a standard mode in that respect. The differences is in extra parameters:

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"library_name"`|string|Name of the custom node library defined in `custom_node_library_config_list`|&check;|
|`"type"`|string|Must be set to `custom`|&check;|
|`"params"`| json object with string values| a list of parameters and their values which could be used in the custom node implementation||

## Using the pipelines <a name="using-pipelines"></a>

Pipelines can use the same API like the models. There are exactly the same calls for running 
the predictions. The request format must match the pipeline definition inputs.

The pipeline configuration can be queried using [gRPC GetModelMetadata](model_server_grpc_api.md#model-metadata) calls and
[REST Metadata](model_server_rest_api.md#model-metadata).
It returns the definition of the pipelines inputs and outputs. 

Similarly, pipelines can be queried for their state using the calls [GetModelStatus](model_server_grpc_api.md#model-status)
and [REST Model Status](model_server_rest_api.md#model-status)

The only difference in using the pipelines and individual models is in version management. In all calls to the pipelines, 
version parameter is ignored. Pipelines are not versioned. Though, they can reference a particular version of the models in the graph.

## Pipelines examples <a name="pipeline-examples"></a>

[Models ensemble](ensemble_scheduler.md)

[Single face analysis with combined models](combined_model_dag.md)

[Multiple vehicles analysis using demultiplexer and custom node](vehicles_analysis_dag.md)

[Optical Character Recognition with custom node pipeline](east_ocr.md)

## Current limitations <a name="current-limitations"></a>

- Models with ["auto" batch size or shape](shape_batch_size_and_layout.md) cannot be referenced in pipeline
- Connected inputs and output for subsequent node models need to exactly match each other in terms of data shape and precision - 
there is no automatic conversion between input/output model precisions or layouts. This limitation can be addressed with a custom node to transform the data as required to match the data format.
- REST requests with no named format (JSON body with one unnamed input) are not supported


## See Also

- [Optimization of Performance](./performance_tuning.md)
