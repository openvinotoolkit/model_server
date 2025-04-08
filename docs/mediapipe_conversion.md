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

### 2. Replacement of inference calculators in graph and subgraphs

We start with basic replacement of inference calculator in graph and subgraphs if needed. Existing configuration could look like:
```
node {
  calculator: "HandLandmarkModelLoader"
  input_side_packet: "MODEL_COMPLEXITY:model_complexity"
  output_side_packet: "MODEL:model"
}
node {
  calculator: "InferenceCalculator"
  input_side_packet: "MODEL:model"
  input_stream: "TENSORS:input_tensor"
  output_stream: "TENSORS:output_tensors"
  options: {
    [mediapipe.InferenceCalculatorOptions.ext] {
      model_path: "mediapipe/modules/holistic_landmark/hand_recrop.tflite"
      delegate {
        xnnpack {}
      }
    }
  }
}
```
This tells us which model is used (`hand_recrop`) and what type of packets are send to inference calculator (`vector\<mediapipe::Tensor\>`). We also need information what are model names inputs. This could be checked f.e. using OVMS logs from model loading or metadata request calls. With that information we would replace that part of a graph with:
```
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "hand_recrop"
      servable_version: "1"
    }
  }
}
node {
  calculator: "OpenVINOInferenceCalculator"
  input_side_packet: "SESSION:session"
  input_stream: "TENSORS:initial_crop_tensor"
  output_stream: "TENSORS:landmark_tensors"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
          tag_to_input_tensor_names {
            key: "TENSORS"
            value: "input_1"
          }
          tag_to_output_tensor_names {
            key: "TENSORS"
            value: "output_crop"
          }
        }
  }
}
```
In `OpenVINOModelServerSessionCalculator` we set `servable_name` with the model's name we found earlier. In `OpenVINOInferenceCalculator` we set input & output tags names to start with `TENSORS`. We then need to map out those tags to actual model names in `mediapipe.OpenVINOInferenceCalculatorOptions` `tag_to_input_tensor_names` and `tag_to_output_tensor_names` fields.

#### 2.1. Add information about input/output tensors ordering.

This step may be required if model has multiple inputs or outputs. If input/output packet types are vector of some type - we must figure out the correct ordering of tensors - expected by the graph. Assuming that model produces several outputs we may need to add following section to `OpenVINOInferenceCalculatorOptions`:
```
output_order_list: ["Identity","Identity_1","Identity_2","Identity_3"]
```
In case of multiple inputs, we must do similar steps, and add:
```
input_order_list: ["Identity","Identity_1","Identity_2","Identity_3"]
```

### 3. Adjust graph input/output streams

This step is required if you plan to deploy the graph in OpenVINO Model Server and existing graph does not have supported input/output packet types. Check for supported input and output packet types [here](./mediapipe.md).
In that cases you may need to add converter calculators as it was done [here](https://github.com/openvinotoolkit/model_server/blob/releases/2025/1/demos/mediapipe/object_detection/graph.pbtxt#L31).

### 4. Set the config.json file path in the session calculator

This step is required if you plan to use graph within existing application. You need to fill `server_config` field in `OpenVINOModelServerSessionCalculatorOptions` to pass full file path to configuration file like:
```
node {
  calculator: "OpenVINOModelServerSessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
      servable_name: "hand_recrop"
      servable_version: "1"
      server_config: "/servables/config.json"
    }
  }
}
```
