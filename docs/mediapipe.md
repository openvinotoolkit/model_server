# Integration with mediapipe {#ovms_docs_mediapipe}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

@endsphinxdirective

## Introduction
MediaPipe is an open-source framework for building pipelines to perform inference over arbitrary sensory data. If comes with a wide range of calculators/nodes which can be applied for unlimited number of scenarios in image and media analytics, generative AI, transformers and many more. Here can be found more information about [MediaPipe framework ](https://developers.google.com/mediapipe/framework/framework_concepts/overview)

Thanks to the integration between Mediapipe and OpenVINO Model server, the graphs can be exposed over the network and the complete load can be delegated to a remote host or a microservice service.
We support the following scenarios:
- stateless execution via unary to unary gRPC calls 
- stateful graph execution via [gRPC streaming](./streaming_endpoints.md) to streaming sessions.

With the introduction of OpenVINO calculator it is possible to optimize inference execution in the OpenVINO Runtime backend. This calculator can be employed both in the graphs deployed inside the Model Server but also in the standalone applications using the Mediapipe framework.
![mp_graph_modes](./mp_graph_modes.png)

Check [this github repository](https://github.com/openvinotoolkit/mediapipe) to learn how to use the OpenVINO calculator inside standalone Mediapipe application. 

This guide gives information about:

* <a href="#ovms-calculators">OVMS Calculators</a>
* <a href="#create-graph">How to create the graph for deployment in OVMS</a>
* <a href="#graph-deploy">Graph deployment</a>
* <a href="#testing">Deployment testing</a>
* <a href="#client">Using MediaPipe graphs from the remote client </a>
* <a href="#updating-graph">How to update existing graphs to use OV for inference </a>
* <a href="#adding-calculator">Adding your own mediapipe calculator to OpenVINO Model Server </a>
* <a href="#graph-examples">Demos and examples</a>
* <a href="#current-limitations">Current Limitations</a>



## OpenVINO calculators <a name="ovms-calculators"></a>

We are introducing a set of calculators which can bring to the graphs execution the advantage of OpenVINO Runtime.

Check their documentation on https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/calculators/ovms/calculators.md


## Python calculator
TBD

## How to create the graph for deployment in OVMS <a name="create-graph"></a>

### Supported graph input/output streams packet types
OpenVINO Model Server supports processing several packet types at the inputs and outputs of the graph.
Following table lists supported tag and packet types based on pbtxt configuration file line:

|pbtxt line|input/output|tag|packet type|stream name|
|:---|:---|:---|:---|:---|
|input_stream: "a"|input|none|ov::Tensor|a|
|input_stream: "IMAGE:a"|input|IMAGE|mediapipe::ImageFrame|a|
|output_stream: "OVTENSOR:b"|output|OVTENSOR|ov::Tensor|b|
|input_stream: "REQUEST:req"|input|REQUEST|KServe inference::ModelInferRequest|req|
|output_stream: "RESPONSE:res"|output|RESPONSE|KServe inference::ModelInferResponse|res|

In case of missing tag OpenVINO Model Server assumes that the packet type is `ov::Tensor'.
For list of supported packet types and tags of OpenVINOInferenceCalculator check documentation of [OpenVINO Model Server calculators](https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/calculators/ovms/calculators.md).

Input serialization to MediaPipe ImageFrame format, requires the data in the KServe request to be encapsulated in `raw_input_contents` field. That is the default behavior in the client libs like `triton-client`.
The required data layout for the Mediapipe Image conversion is HWC and the supported precisions are:
|Datatype|Allowed number of channels|
|:---|:---|
|FP16|1,3,4|
|FP32|1,2|
|UINT8|1,3,4|
|INT8|1,3,4|
|UINT16|1,3,4|
|INT16|1,3,4|

When the client is sending in the gRPC request the input as an numpy array, it will be deserialized on the Model Server side to the format specified in the graph.
For example when the graph has the input type IMAGE, the gRPC client could send the input data with the shape (300, 300, 3) and precision INT8. It would not be allowed to send the data in the shape for example (1,300,300,1) as that would be incorrect layout and the number of dimensions.

When the input graph would be set as OVTENSOR, an arbitrary shape and precisions on the input would be allowed. It will be converted to OV::Tensor object and passed to the graph. For example input with shape (1,3,300,300) FP32 assuming that format would be accepted by the graph calculators.

There is also an option to avoid any data conversions in the serialization and deserialization by the OpenVINO Model Server. When the input stream is of type REQUEST, it will be passed through to the calculator. The receiving calculator will be in charge to deserialize it and interpret all the content. Likewise, the output format RESPONSE delegate to the calculator creating a complete KServe response message to the client. That gives extra flexibility in the data format.

### Side packets
Side packets are special parameters which can be passed to the calculators at the beginning of the graph initialization. It can tune the behavior of the calculator like set the object detection threshold or number of objects to process.
With KServe gRPC API you are also able to push side input packets into graph. They are to be passed as KServe request parameters. They can be of type string, int64 or boolean.
Note that with the gRPC stream connection, only the first request in the stream can include the side package parameters.


Review also an example in [object detection demo](../demos/mediapipe/object_detection/README.md)

### List of default calculators
Beside OpenVINO inference calculators, there are included, by default, in the public image also all the calculators used in the enabled demos. 
They can be identified in the bazel [BUILD](../src/BUILD) file in the target `ovms_lib` directly or in the nested dependencies in the included support for mediapipe graphs like:
[holistic](https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/graphs/holistic_tracking/BUILD),
[object detection](https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/graphs/object_detection/BUILD) etc.

TBD how the list calculators with bazel cmd


### CPU and GPU execution
As of now, the calculators included in the public docker images supports only CPU execution. They exchange between nodes objects and memory buffers form the host memory. While the GPU buffers are not supported in the mediapipe graphs, it is still possible to run the inference operation on GPU target device.
The input data and the response will be automatically exchanges between GPU and host memory. For that scenarios, make sure to use the build image with `-gpu` suffix in the tag as it includes required dependencies for Intel discrete and integrated GPU.

Full pipeline execution on the GPU is expected to be added in future releases.

## Graph deployment <a name="graph-deploy"></a>
### How to package the graph and models
in order to simplify distribution of the graph artefacts and the deploymment process in various environments,
it is recommended to create a specific folders structure:
```bash
mediapipe_graph_name/
├── graph.pbtxt
├── model1
│   └── 1
│       ├── model.bin
│       └── model.xml
├── model2
│   └── 1
│       ├── model.bin
│       └── model.xml
└── subconfig.json
```

The `graph.pbtxt` should include the definition of the mediapipe graph. 
The `subconfig.json` is an extension to the main model server `config.json` configuration file. The subconfig should include the definition of the models used in the graph nodes. They will be loaded during the model server intialization and ready to use when the graph starts executing.
Included OpenVINO inference session caclulator should include the reference to the model name as configured in `subconfig.json`.
Here is example of the `subconfig.json`:
```json
{
    "model_config_list": [
        {"config": {
                "name": "model1_name",
                "base_path": "model1"
        	  }
	      }
    ]
}
```


### Starting OVMS with MP servables
MediaPipe servables configuration is to be placed in the same json file like the 
[models config file](starting_server.md).
While models are defined in section `model_config_list`, graphs are configured in
the `mediapipe_config_list` section. 

While the mediapipe graphs artifacts are packaged like presented above, configuring the OpenVINO Model Server is easy. Just a `config.json` needs to be prepared list all the graphs for deployments:
```json
{
    "model_config_list": [],
    "mediapipe_config_list": [
    {
        "name":"mediapipe_graph_name"
    },
    {
        "name":"mediapipe2",
        "base_path":"non_default_path"
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


## Deployment testing <a name="testing"></a>
### Debug logs
The simples method to validate the graph execution is to set the Model Server `log_level` do `DEBUG`.
`docker run --rm -it -v $(pwd):/config openvino/model_server:latest --config_path /config/config.json --log_level DEBUG`

It will report in a verbose way all the operations in the mediapipe framework from the graph initialization and also the graphs execution.
After the model server, you could confirm with the graph has correct format and all the required models are loaded successfully.
Note that graph loading is not confirming if all the calculators are compiled into the model server build. That can be confirmed with after sending the request to the KServe endpoint.
During the requests processing, the logs will include info about calculators registration and processing the nodes.

### Tracing
Currently the graph tracing on the model server side is not supported. If you would like to take advantage of mediapipe tracing to identify the graph bottleneck, test the graph from the mediapipe application level. Build an example application similarly to [holistic app](https://github.com/openvinotoolkit/mediapipe/tree/main/mediapipe/examples/desktop/holistic_tracking) with the steps documented on [mediapipe tracking](https://github.com/google/mediapipe/blob/master/docs/tools/tracing_and_profiling.md).

### Benchmarking
While you implemented and deployed the graph you have several options to test the performance.
To validate the throughput for unary requests you can use the [benchmark client](../demos/benchmark/python#mediapipe-benchmarking).

For streaming gRPC connections, there is available [rtps_client](https://github.com/openvinotoolkit/model_server/tree/llama-ovms/demos/mediapipe/holistic_tracking#rtsp-client).
It can generate the load to gRPC stream and the mediapipe graph based on the content from RTSP video stream, MPG4 file or from the local camera.

## Using MediaPipe graphs from the remote client <a name="client"></a>

MediaPipe graphs can use the same gRPC KServe Inference API both the the unary calls and the streaming. 
The same client libraries with KServe API support can be used in both cases. The client code for the unary and streaming is a bit different.
Check the [code snippets](https://docs.openvino.ai/2023.2/ovms_docs_clients_kfs.html)

Review also the information about the [gRPC streaming feature](./streaming_endpoints.md)

Graphs can be queried for their state using the calls [GetModelStatus](model_server_grpc_api_kfs.md)
and [REST Model Status](model_server_rest_api_kfs.md)

The difference in using the mediapipes and individual models is in version management. In all calls to the mediapipes,
the version parameter is ignored. Mediapipes are not versioned. Though, they can reference a particular version of the models in the graph.



## How to update existing graphs to use OV for inference <a name="updating-graph"></a>

If you would like to reusing existing graph and replace Tensorflow execution with OpenVINO backend, check this [guide](TBD)

## Adding your own mediapipe calculator to OpenVINO Model Server <a name="adding-calculator"></a>
MediaPipe graphs can include only the calculators built-in the model server during the image build.
If you want to add your own mediapipe calculator to OpenVINO Model Server functionality you need to add it as a dependency and rebuild the OpenVINO Model Server binary.

If you have it in external repository, you need to add the http_archive() definition or git_repository() definition to the bazel [WORKSPACE](../WORKSPACE) file.
Then you need to add the calculator target as a bazel dependency to the [src/BUILD](../src/BUILD) file. This should be done for:

```
cc_library(
 name = "ovms_lib",
...
```

in the conditions:default section of the deps property:

```
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


## MediaPipe Graphs Examples <a name="graphs-examples"></a>

[Holistic analysis](../demos/mediapipe/holistic_tracking)

[Image classification](../demos/mediapipe/image_classification/README.md)

[Object detection](../demos/mediapipe/object_detection)

[Multi model](../demos/mediapipe/multi_model_graph/README.md)



## Current limitations <a name="current-limitations"></a>
- MediaPipe graphs are supported only for gRPC KServe API.

- KServe ModelMetadata call response contains only input and output names. In the response shapes will be empty and datatypes will be `"INVALID"`.

- Binary inputs are not supported for MediaPipe graphs.

- Updates in subconfig files and mediapipe graph files do not trigger model server config reloads. The reload of the full config, including subconfig and graphs, can be initiated by an updated in the main config json file or using the REST API `config/reload` endpoint. 


