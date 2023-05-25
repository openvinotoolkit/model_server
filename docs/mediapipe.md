# Mediapipe {#ovms_docs_mediapipe}

## Introduction
MediaPipe is a production ready framework for building pipelines to perform inference over arbitrary sensory data. Using MediaPipe in the OVSM enables user to define a powerful graph from a lot of ready calculators/nodes that come with the MediaPipe which support all the needed features for running a stable graph like e.g. flow limiter node. User can also run the graph in a server or run it inside application host. Here can be found more information about [MediaPipe framework ](https://developers.google.com/mediapipe/framework/framework_concepts/overview)

This guide gives information about:

* <a href="#how-to-build">How to build OVMS with MediaPipe support</a>
* <a href="#ovms-calculators">OVMS Calculators</a>
* <a href="#graph-proto">Graph proto files</a>
* <a href="#configuration-files">Configuration files</a>
* <a href="#using-mediapipe">Using the mediapipe graphs</a>
* <a href="#graphs-examples">Graphs examples </a>
* <a href="#current-limitations">Current Limitations</a>

## How to build OVMS with mediapipe support <a name="how-to-build"></a>
Building OVMS with mediapipe support requires passing additional flag for make command, for example:

```
MEDIAPIPE_DISABLE=0 make docker_build
```

## Node Types <a name="ovms-calculators"></a>

"Each calculator is a node of a graph. The bulk of graph execution happens inside its calculators. Ovms has its own calculators but can also use newly developed calculators or reuse the existing calculators defined in the oryginal mediapipe repository."

For more details you can visit mediapipe concept description - [Calculators Concept Page](https://developers.google.com/mediapipe/framework/framework_concepts/calculators) or Ovms specific calculators implementation - [Ovms Calculators Concept Page](https://github.com/openvinotoolkit/model_server/blob/releases/2023/0/src/mediapipe_calculators/calculators.md)

## Graph proto files <a name="graph-proto"></a>

Graph proto files are used to define a graph. Example content of graph containing OVMSCalculator node proto file:

```

input_stream: "in"
output_stream: "out"
node {
  calculator: "OVMSOVCalculator"
  input_stream: "B:in"
  output_stream: "A:out"
  node_options: {
        [type.googleapis.com / mediapipe.OVMSCalculatorOptions]: {
          servable_name: "dummy"
          servable_version: "1"
          tag_to_input_tensor_names {
            key: "B"
            value: "b"
          }
          tag_to_output_tensor_names {
            key: "A"
            value: "a"
          }
        }
  }
}


Here can be found more information about [MediaPipe graphs proto](https://developers.google.com/mediapipe/framework/framework_concepts/graphs)

```

## Configuration files <a name="configuration-files"></a>
MediaPipe graph configuration is to be placed in the same json file like the 
[models config file](starting_server.md).
While models are defined in section `model_config_list`, graphs are configured in
the `mediapipe_config_list` section. 
Nodes in the MediaPipe graphs can reference both to the models configured in model_config_list section and in subconfig.

Basic graph section template is depicted below:

```

{
    "model_config_list": [...],
    "mediapipe_config_list": [
    {
        "name":"mediaDummy",
        "base_path":"/mediapipe/graphs/",
        "graph_path":"graphdummyadapterfull.pbtxt",
        "subconfig":"subconfig_dummy.json"
    }
    ]
}

```

Basic subconfig:

```

{
    "model_config_list": [
        {"config": {
                "name": "dummy",
                "base_path": "/models/dummy",
                "shape": "(1, 10)"
        	}
	}
    ]
}


```


### MediaPipe configuration options explained

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"name"`|string|Graph identifier related to name field specified in gRPC/REST request|Yes|
|`"base_path"`|string|Path to the which graph definition and subconfig files paths are relative. May be absolute or relative to the main config path. Default value is "{main config path}\"|No|
|`"graph_path"`|string|Path to the graph proto file. May be absolute or relative to the base_path. Default value is "{base_path}\graph.pbtxt". File have to exist.|No|
|`"subconfig"`|string|Path to the subconfig file. May be absolute or relative to the base_path. Default value is "{base_path}\subconfig.json". Non existence of file does not result in error.|No|

Subconfig file may contain only one section wich is model_config_list the same thats in [models config file](starting_server.md).

## Using Mediapipe <a name="using-mediapipe"></a>

MediaPipe graphs can use the same API as the models. There are exactly the same calls for running 
the predictions. The request format must match the pipeline definition inputs.

The graph configuration can be queried using [gRPC GetModelMetadata](model_server_grpc_api_tfs.md) calls and
[REST Metadata](model_server_rest_api_tfs.md).
It returns the definition of the graphs inputs and outputs. 

Similarly, graphs can be queried for their state using the calls [GetModelStatus](model_server_grpc_api_tfs.md)
and [REST Model Status](model_server_rest_api_tfs.md)

## MediaPipe Graps Examples <a name="graphs-examples"></a>


## Current limitations <a name="current-limitations"></a>

- Making changes in subconfig file does not trigger config reloads even if file_system_poll_wait_seconds parameter value was different than 0 during OVMS start. Only main config changes are monitored.
