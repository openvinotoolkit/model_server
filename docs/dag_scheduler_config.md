# DAG Configuration File  {#ovms_docs_dag_config}

* <a href="#configuration-file">Configuration file</a>
* <a href="#current-limitations">Current Limitations</a>


## Configuration file <a name="configuration-file"></a>

Pipelines configuration is to be placed in the same json file like the 
[models config file](multiple_models_mode.md).
While models are defined in section `model_config_list`, pipelines are to be configured in
section `pipeline_config_list`. 
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


## Current limitations <a name="current-limitations"></a>

- Models with "auto" [batch size](dynamic_bs_auto_reload.md) or [shape](dynamic_shape_auto_reload.md) cannot be referenced in pipeline
- Connected inputs and output for subsequent node models need to match each other in terms of data shape, precision and layout - 
there is no automatic conversion between input/output model precisions or layouts. This limitation can be addressed with `--shape` and `--layout` model configuration or with a custom node to transform the data as required to match the expected data format.
- REST requests with no named format (JSON body with one unnamed input) are not supported


## See Also

- [Optimization of Performance](./performance_tuning.md)
