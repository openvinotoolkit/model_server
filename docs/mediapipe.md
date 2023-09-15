# Integration with mediapipe {#ovms_docs_mediapipe}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

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
* <a href="#adding-calculator">Adding calculator</a>


## Node Types <a name="ovms-calculators"></a>

"Each calculator is a node of a graph. The bulk of graph execution happens inside its calculators. OVMS has its own calculators but can also use newly developed calculators or reuse the existing calculators defined in the original mediapipe repository."

For more details you can visit mediapipe concept description - [Calculators Concept Page](https://developers.google.com/mediapipe/framework/framework_concepts/calculators) or OVMS specific calculators implementation - [Ovms Calculators Concept Page](https://github.com/openvinotoolkit/model_server/blob/releases/2023/0/src/mediapipe_calculators/calculators.md)

## Graph proto files <a name="graph-proto"></a>

Graph proto files are used to define a graph. Example content of proto file with graph containing OpenVINO inference nodes:

```

input_stream: "in1"
input_stream: "in2"
output_stream: "out"
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:dummy"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "dummy"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:add"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "add"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:dummy"
  input_stream: "DUMMY_IN:in1"
  output_stream: "DUMMY_OUT:dummy_output"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
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
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:add"
  input_stream: "ADD_INPUT1:dummy_output"
  input_stream: "ADD_INPUT2:in2"
  output_stream: "SUM:out"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
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

# Supported input/output packet types

OVMS does support processing several packet types at the inputs and outputs of the graph.
Following table lists supported tag and packet types based on pbtxt configuration file line:

|pbtxt line|input/output|tag|packet type|stream name|
|:---|:---|:---|:---|:---|
|input_stream: "a"|input|none|ov::Tensor|a|
|input_stream: "IMAGE:a"|input|IMAGE|mediapipe::ImageFrame|a|
|output_stream: "OVTENSOR:b"|output|OVTENSOR|ov::Tensor|b|
|input_stream: "REQUEST:req"|input|REQUEST|KServe inference::ModelInferRequest|req|
|output_stream: "RESPONSE:res"|output|RESPONSE|KServe inference::ModelInferResponse|res|

In case of missing tag OVMS assumes that the packet type is `ov::Tensor'.
For list of supported packet types and tags of OpenVINOInferenceCalculator check documentation of [OVMS calculators](https://github.com/openvinotoolkit/model_server/blob/main/src/mediapipe_calculators/calculators.md).

With KServe gRPC API you are also able to push side input packets into graph. In this case created side packet type is the same as KServe parameter type (string, int64 or boolean).

`Image` inputs requires image pixel data inside raw_input_contents that can be converted to MediaPipe ImageFrame format. For now, those kind of inputs only accepts three-dimensional data in HWC layout. Datatypes supported for `Image` format:
|Datatype|Number of channels|
|:---|:---|
|FP16|1,3,4|
|FP32|1,2|
|UINT8|1,3,4|
|INT8|1,3,4|
|UINT16|1,3,4|
|INT16|1,3,4|


Documentation on handling tags inside OVMS calculators is placed [here](https://github.com/openvinotoolkit/mediapipe/blob/237bf3bf23c4d7b7e38eb92b4a3b6d540d83421b/mediapipe/calculators/ovms/calculators.md).

## Configuration files <a name="configuration-files"></a>
MediaPipe pipelines configuration is to be placed in the same json file like the 
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
|`"base_path"`|string|Path to the which graph definition and subconfig files paths are relative. May be absolute or relative to the main config path. Default value is "(main config path)\(name)"|No|
|`"graph_path"`|string|Path to the graph proto file. May be absolute or relative to the base_path. Default value is "(base_path)\graph.pbtxt". File have to exist.|No|
|`"subconfig"`|string|Path to the subconfig file. May be absolute or relative to the base_path. Default value is "(base_path)\subconfig.json". Missing  file does not result in error.|No|

Subconfig file may only contain *model_config_list* section  - in the same format as in [models config file](starting_server.md).

## Using MediaPipe <a name="using-mediapipe"></a>

MediaPipe graphs can use the same KServe Inference API as the models. There are exactly the same calls for running
the predictions. The request format must match the pipeline definition inputs.

Graphs can be queried for their state using the calls [GetModelStatus](model_server_grpc_api_kfs.md)
and [REST Model Status](model_server_rest_api_kfs.md)


## MediaPipe Graphs Examples <a name="graphs-examples"></a>

[Image classification](../demos/mediapipe/image_classification/README.md)

[Multi model](../demos/mediapipe/multi_model_graph/README.md)

## Current limitations <a name="current-limitations"></a>
- MediaPipe graphs are supported only for gRPC KServe API.

- KServe ModelMetadata call response contains only input and output names. In the response shapes will be empty and datatypes will be `"INVALID"`.

- Binary inputs are not supported for MediaPipe graphs.

- Making changes in subconfig file does not trigger config reloads. Main config changes are monitored and triggers subconfig reload even if this wasn't changed. Changes in main config json trigger also checking for changes in graph's pbtxt files.

## Adding your own mediapipe calculator to OVMS <a name="adding-calculator"></a>
If you want ot add your own mediapipe calculator to OVMS functionality you need to add it as a dependency and rebuild the ovms binary.

If you have it in external repository, you need to add the http_archive() definition or git_repository() definition to the bazel WORKSPACE file.
Then you need to add the calculator target as a bazel dependency to the src/BUILD file. This should be done for:
cc_library(
 name = "ovms_lib",
...

in the conditions:default section of the deps property:

```bash
  deps = [
         "//:ovms_dependencies",
        "//src/kfserving_api:kfserving_api_cpp",
        ] + select({
            "//conditions:default": [
                "//src:ovmscalculatoroptions_cc_proto", # ovmscalculatoroptions_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
                "@mediapipe_calculators//:mediapipe_calculators",
                 "@your_repository//:yourpathtocalculator/your_calculator
```

 Make sure the REGISTER_CALCULATOR(your_calculator); macro is present in the calculator file that you have added.
