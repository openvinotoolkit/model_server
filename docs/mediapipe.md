# Integration with mediapipe (preview) {#ovms_docs_mediapipe}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_mediapipe_calculators

@endsphinxdirective

## Introduction
MediaPipe is an open-source framework for building pipelines to perform inference over arbitrary sensory data. Using MediaPipe in the OVMS enables user to define a powerful graph from a lot of ready calculators/nodes that come with the MediaPipe which support all the needed features for running a stable graph like e.g. flow limiter node. User can also run the graph in a server or run it inside application host. Here can be found more information about [MediaPipe framework ](https://developers.google.com/mediapipe/framework/framework_concepts/overview)

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

More information about OVMS build parameters can be found here [here](https://github.com/openvinotoolkit/model_server/blob/develop/docs/build_from_source.md).

## Node Types <a name="ovms-calculators"></a>

"Each calculator is a node of a graph. The bulk of graph execution happens inside its calculators. OVMS has its own calculators but can also use newly developed calculators or reuse the existing calculators defined in the original mediapipe repository."

For more details you can visit mediapipe concept description - [Calculators Concept Page](https://developers.google.com/mediapipe/framework/framework_concepts/calculators) or OVMS specific calculators implementation - [Ovms Calculators Concept Page](https://github.com/openvinotoolkit/model_server/blob/releases/2023/0/src/mediapipe_calculators/calculators.md)

## Graph proto files <a name="graph-proto"></a>

Graph proto files are used to define a graph. Example content of proto file with graph containing ModelAPICalculator nodes:

```

input_stream: "in1"
input_stream: "in2"
output_stream: "out"
node {
  calculator: "ModelAPISessionCalculator"
  output_side_packet: "SESSION:dummy"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPIOVMSSessionCalculatorOptions]: {
      servable_name: "dummy"
      servable_version: "1"
    }
  }
}
node {
  calculator: "ModelAPISessionCalculator"
  output_side_packet: "SESSION:add"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPIOVMSSessionCalculatorOptions]: {
      servable_name: "add"
      servable_version: "1"
    }
  }
}
node {
  calculator: "ModelAPISideFeedCalculator"
  input_side_packet: "SESSION:dummy"
  input_stream: "DUMMY_IN:in1"
  output_stream: "DUMMY_OUT:dummy_output"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPIInferenceCalculatorOptions]: {
        tag_to_input_tensor_names {
          key: "DUMMY_IN"
          value: "b"
        }
        tag_to_output_tensor_names {
          key: "DUMMY_OUT"
          value: "a"
        }
    }
  }
}
node {
  calculator: "ModelAPISideFeedCalculator"
  input_side_packet: "SESSION:add"
  input_stream: "ADD_INPUT1:dummy_output"
  input_stream: "ADD_INPUT2:in2"
  output_stream: "SUM:out"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPIInferenceCalculatorOptions]: {
        tag_to_input_tensor_names {
          key: "ADD_INPUT1"
          value: "input1"
        }
        tag_to_input_tensor_names {
          key: "ADD_INPUT2"
          value: "input2"
        }
        tag_to_output_tensor_names {
          key: "SUM"
          value: "sum"
        }
    }
  }
}

```


Here can be found more information about [MediaPipe graphs proto](https://developers.google.com/mediapipe/framework/framework_concepts/graphs)

## Configuration files <a name="configuration-files"></a>
MediaPipe graph configuration is to be placed in the same json file like the 
[models config file](starting_server.md).
While models are defined in section `model_config_list`, graphs are configured in
the `mediapipe_config_list` section. 

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
Nodes in the MediaPipe graphs can reference both to the models configured in model_config_list section and in subconfig.

### MediaPipe configuration options explained

|Option|Type|Description|Required|
|:---|:---|:---|:---|
|`"name"`|string|Graph identifier related to name field specified in gRPC/REST request|Yes|
|`"base_path"`|string|Path to the which graph definition and subconfig files paths are relative. May be absolute or relative to the main config path. Default value is "(main config path)\"|No|
|`"graph_path"`|string|Path to the graph proto file. May be absolute or relative to the base_path. Default value is "(base_path)\graph.pbtxt". File have to exist.|No|
|`"subconfig"`|string|Path to the subconfig file. May be absolute or relative to the base_path. Default value is "(base_path)\subconfig.json". Missing  file does not result in error.|No|

Subconfig file may only contain *model_config_list* section  - in the same format as in [models config file](starting_server.md).

## Using Mediapipe <a name="using-mediapipe"></a>

MediaPipe graphs can use the same KServe Inference API as the models. There are exactly the same calls for running
the predictions. The request format must match the pipeline definition inputs.

Graphs can be queried for their state using the calls [GetModelStatus](model_server_grpc_api_kfs.md)
and [REST Model Status](model_server_rest_api_kfs.md)

## MediaPipe Graphs Examples <a name="graphs-examples"></a>

[Image classification](../demos/mediapipe/image_classification/README.md)

[Multi model](../demos/mediapipe/multi_model_graph/README.md)

## Current limitations <a name="current-limitations"></a>
- It is preview version of the MediaPipe integrations which means that its not ready to be used in production and only some of the OVMS features are supported for Mediapipe graphs.

- Mediapipe graphs are supported only for GRPC KFS API. Only TFS calls supported are get model status and config reload.

- Binary inputs are not supported for MediaPipe graphs.

- Public images do not include mediapipe feature.

- Making changes in subconfig file does not trigger config reloads. Main config changes are monitored and triggers subconfig reload even if those weren't changed.
