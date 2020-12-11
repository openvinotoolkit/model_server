# Directed Acyclic Graph Scheduler in OpenVINO&trade; Model Server

## Introduction
OpenVINO&trade; Model Server provides possibility to create pipeline of models for execution in a single client request. 
Pipeline is a Directed Acyclic Graph with different nodes which define how to process each step of predict request. 
By using such pipeline, there is no need to return intermediate results of every model to the client. This allows avoiding the network overhead by minimizing the number of requests sent to model server. 
Each model output can be mapped to another model input. Since intermediate results are kept in server's RAM these can be reused by subsequent inferences which reduces overall latency.

This guide gives information about following:

* <a href="#node-type">Node Types</a>
    * Pre-defined Node Types
    * Other Node Types
* <a href="#example">Example Use Case</a>
    * Prepare the models
    * Define a pipeline
    * Start model server
    * Requesting the service
    * Analyze pipeline execution in server logs
    * Requesting pipeline metadata


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

## Configuration file

Pipelines configuration is to be placed in the same json file like the 
[models config file](docker_container.md#configfile).
While models are defined in section `model_config_list`, pipelines are to be configured in
section `pipeline_config_list`. 
Nodes in the pipelines can reference only the models configured in model_config_list section.

Below is depicted a basic pipeline section template:

```
{
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



## Pipeline configuration options explained

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


## Using the pipelines

Pipelines can use the same API like the models. There are exactly the same calls for running 
the predictions. The request format much much the pipeline definition inputs.

The pipeline configuration can be queried using [gRPC GetModelMetadata](model_server_grpc_api.md#model-metadata) calls and
[REST Metadata](model_server_rest_api.md#model-metadata).
It returns the definition of the pipelines inputs and outputs. 

Similarly, pipelines can be queried for their state using the calls [GetModelStatus](model_server_grpc_api.md#model-status)
and [REST Model Status](model_server_rest_api.md#model-status)

The only difference in using the pipelines and individual models is in version management. In all calls to the pipelines, 
version parameter is ignored. Pipelines are not versioned. Though, they can reference a particular version of the models in the graph.

## Pipelines examples

[Models ensemble](ensemble_scheduler.md)

[Face analysis with combined models](combined_model_dag.md)



## Current limitations

- Models with ["auto" batch size or shape](shape_and_batch_size.md) cannot be referenced in pipeline
- Connected inputs and output for subsequent node models need to exactly match each other in terms of data shape and precision - 
there is no automatic conversion between input/output model precisions or layouts
- REST requests with no named format (JSON body with one unnamed input) are not supported


## See Also

- [Optimization of Performance](./performance_tuning.md)