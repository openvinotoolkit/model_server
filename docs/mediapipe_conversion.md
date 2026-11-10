# How to update existing graphs from MediaPipe framework to use OpenVINO for inference {#ovms_docs_mediapipe_conversion}
In this document we will walkthrough steps required to update existing Mediapipe graphs using Tensorflow/TfLite to make them use OpenVINO Runtime for the inference. The step will include:
- retrieving models from existing solutions
- prepare configuration of OpenVINOInferenceSession calculator
- make changes to existing pbtxt graphs to replace TensorFlow calculators with OpenVINO calculators

## How to get models used in MediaPipe solutions
When you build MediaPipe applications or solutions from the [https://github.com/google/mediapipe](https://github.com/google/mediapipe) repo, typically the bazel build would download the needed models as a data dependency. When the graph is to be deployed with OpenVINO Inference calculators, the models needs to be stored in the [models repository](models_repository.md).
That way you can take advantage of the [models versioning feature](./model_version_policy.md) and store the models on the local or the [cloud storage](./using_cloud_storage.md). The OpenVINO calculator is using as a parameter the path to the [config.json](starting_server.md#serving-multiple-models) file with models configuration with the specific model name.
To get the model used in MediaPipe demo you can either trigger the original build target that depends upon that model and then search in bazel cache or download directly from locations below.
* https://storage.googleapis.com/mediapipe-models/
* https://storage.googleapis.com/mediapipe-assets/

## How to prepare the configuration for the OpenVINO Model Server
We must prepare OVMS [configuration files](starting_server.md) and [models repository](models_repository.md). There are two ways that would have different benefits:
1. First one is recommended if you reuse models between several pipelines in the same deployment. In this case servables directory structure would look like:
```
servables/
├── config.json
├── add_two_inputs_model
│   └── 1
│       ├── add.bin
│       └── add.xml
├── dummy
│   └── 1
│       ├── dummy.bin
│       └── dummy.xml
└── dummyAdd
    └── graph.pbtxt
```
And the config.json:
```
{
  "model_config_list": [
    {
      "config": {
        "name": "dummy",
        "base_path": "dummy"
      }
    },
    {
      "config": {
        "name": "add",
        "base_path": "add_two_inputs_model"
      }
    }
  ]
  "mediapipe_config_list": [
    {
      "name":"dummyAdd"
    }
  ]
}
```
2. Second would be better if you would have several services each containing separate mediapipe. This way can make for easier updates to the deployments and keep mediapipes configurations self-contained. In this case you would prepare directories as shown below
```
servables/
├── config.json
└── dummyAddGraph
    ├── add_two_inputs_model
    │   └── 1
    │       ├── add.bin
    │       └── add.xml
    ├── dummy
    │   └── 1
    │       ├── dummy.bin
    │       └── dummy.xml
    ├── graph.pbtxt
    └── subconfig.json
```
and config.json:
```
{
  "model_config_list": [],
  "mediapipe_config_list": [
    {
      "name":"dummyAddGraph"
    }
  ]
}
```
and the subconfig.json:
```
{
  "model_config_list": [
    {
      "config": {
        "name": "dummy",
        "base_path": "dummy"
      }
    },
    {
      "config": {
        "name": "add",
        "base_path": "add_two_inputs_model"
      }
    }
  ]
}
```
You can find more details about OpenVINO Model Server configuration in [documentation](starting_server.md#serving-multiple-models).

*Note*: base paths in config.json are relative to the file path of config.json.

Now we have configuration for OpenVINO Model Server.

## How to adjust existing graphs to perform inference with OpenVINO Model Server

Below are presented steps to adjust existing graph using TensorFlow or TensorFlowLite calculators to use OpenVINO as the inference engine. That way the inference execution can be optimized while preserving the overall graph structure.

### 1. Identify all inference calculators used in nested subgraphs.

This steps is not needed if there are no subgraphs. Let's assume we start with graph like [this](https://github.com/google/mediapipe/blob/v0.10.3/mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt).
We cannot find direct usage of inference calculators in this graph and that is because it is using `subgraph` concept from MediaPipe framework. It allows you to register existing graph as a single calculator. We must search for such nodes in graph and find out each subgraph that is directly using inference calculators. We can grep the MediaPipe code for:
```
grep -R -n "register_as = \"HolisticLandmarkCpu"
```
We will find that in using bazel `mediapipe_simple_subgraph` function another `pbtxt` file was registered as a graph. Since in that file there is no inference calculator we need to repeat the procedure until we find all inference calculators used directly or indirectly using subgraphs.

After performing those steps we have a list of pbtxt files with inference nodes that need adjustments.

### 2. Use the OpenVINO MediaPipe calculators

OVMS uses `OpenVINOSessionCalculator` and `OpenVINOInferenceCalculator` from the OpenVINO MediaPipe calculator package. The session compiles a model file and publishes a `SESSION` side packet; inference consumes and emits vectors of `ov::Tensor`:

```protobuf
node {
  calculator: "OpenVINOSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {
      model_path: "/models/hand_recrop/1/hand_recrop.xml"
      device: "CPU"
      num_infer_requests: 4
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "TENSORS:input_tensors"
  output_stream: "TENSORS:output_tensors"
  node_options: {
    [type.googleapis.com/mediapipe.OpenVINOInferenceCalculatorOptions] {
      input_order: "input_1"
      output_order: "output_crop"
    }
  }
}
```

`input_order` and `output_order` map vector positions to OpenVINO tensor names. When omitted, tensors bind by model index and outputs follow model order. Use the same `OpenVINOSessionCalculatorOptions` for graphs that should share a compiled model and request pool.

The session calculator reads `model_path` directly. Absolute paths are recommended in deployed graphs. Relative paths resolve from the process working directory, not from the graph or subconfig directory. A session is shared while at least one graph owns it; replacing a model file does not invalidate a live session. Recreate/reload the graph to pick up changed model contents.

The old OVMS calculators selected a model by `servable_name` and version and could start an embedded server using `server_config`. The direct OpenVINO calculators do neither: configure device, request count, shape, layout, and preprocessing in the session options, and manage model files independently of OVMS model configuration. Dynamic input shapes must be declared/compiled through the session options or in the model; the calculator does not reload a servable for each input shape.

### 3. Adapt graph tensor boundaries

The inference calculator accepts `std::vector<ov::Tensor>` on `TENSORS`, `std::vector<ov::RemoteTensor>` on `REMOTE_TENSORS`, or individually tagged `ov::Tensor` streams mapped to model names. Use named streams when inputs come from separate branches; no pack/split calculators are required. Multiple output tags may map to the same model output for fan-out. Vector mode remains available for producers that naturally emit a vector. OVMS registers `MPTensorToOVTensorCalculator` and `OVTensorToMPTensorCalculator` only for graph nodes whose packet type is MediaPipe `Tensor`; those conversions copy data into an owned buffer.

An image or Python producer must produce the packet type expected by its boundary calculator. Check preprocessing/layout/precision when replacing model-server preprocessing with session options; do not assume that changing a stream tag changes its packet type.
