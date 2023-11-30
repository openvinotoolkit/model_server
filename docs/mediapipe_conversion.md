# How to deploy existing graphs from MediaPipe framework with OpenVINO inference calculator {#ovms_docs_mediapipe_conversion}
In this document we will walkthrough steps required to use OVMS with Mediapipe for existing calculators using Tensorflow/TfLite for inference.
## How to get models used in MediaPipe demos
When you build mediapipe applications or solutions, typically the bazel configuration would download the needed models as a dependency. When the graph is to be deployed via the OpenVINO Model Server or when the mediapipe application would use OpenVINO Model Server as the inference executor, the models needs to be stored in the [models repository](https://docs.openvino.ai/2023.2/ovms_docs_models_repository.html).
That way you can take advantage of the model versioning feature and store the models on the local or the cloud storage. The OpenVINO calculator is using as a parameter the path to the [config.json](https://docs.openvino.ai/2023.2/ovms_docs_serving_model.html#serving-multiple-models) file with models configuration with the specific model name.
To get the model used in MediaPipe demo you can either trigger build target that depends upon that model and then search in bazel cache or download directly from locations below.
* https://storage.googleapis.com/mediapipe-models/
* https://storage.googleapis.com/mediapipe-assets/

## How to prepare OpenVINO Model Server deployment with Mediapipe
We must prepare OVMS configuration files and models repository. There are two ways that would have different benefits:
1. First one would be better if you want to have just one model server service containing all servables. This may be especially useful if you will reuse models between several pipelines in the same deployment. In this case servables directory structure would look like:
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
└── dummyAdd
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
      "name":"dummyAdd"
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
In both cases `servables` directory will be mounted to OVMS container. You can find more details about OVMS configuration in [documentation](https://docs.openvino.ai/2023.1/ovms_docs_serving_model.html#serving-multiple-models).

*Note*: base paths in config.json are relative to the file path of config.json.

## How to adjust existing graphs to perform inference with OpenVINO Model Server
Below are presented steps to adjust existing graph using TensorFlow or TensorFlowLite calculators to use OpenVINO as the inference engine. That way the inference execution can be optimized while preserving the overall graph structure.

### 1. Identify all used subgraphs with inference calculators. This steps is not needed if there are no subgraphs.

Let's assume we start with graph like [this](https://github.com/google/mediapipe/blob/v0.10.3/mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt).
We can't find direct usage of inference calculators in this graph and that is because it is using `subgraph` concept from MediaPipe framework. It allows you to register existing graph as a single calculator. We must search for such nodes in graph and find out each subgraph that is directly using inference calculators. We can grep the MediaPipe code for:
```
grep -R -n "register_as = \"HolisticLandmarkCpu"
```
We will find that in using bazel `mediapipe_simple_subgraph` function another `pbtxt` file was registered as a graph. Since in that file there is no inference calculator we need to repeat the procedure until we find all inference calculators used directly or indirectly using subgraphs.
### 2. We need to start with basic replacement of inference calculator.
Existing configuration could look like:
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

### 3. Third step may be required if model has multiple inputs or outputs.
If input/output packet types are vector of some type - we must figure out the correct ordering of tensors - expected by the graph. Assuming that model produces several outputs we may need to add following section to `OpenVINOInferenceCalculatorOptions`:
```
output_order_list: ["Identity","Identity_1","Identity_2","Identity_3"]
```
In case of multiple inputs, we must do similar steps, and add:
```
input_order_list: ["Identity","Identity_1","Identity_2","Identity_3"]
```

### 4. Fourth step may be required if existing graph does not have supported input/output packet types
Check for supported packet types in OVMS [here](./mediapipe.md).
In that cases you may need to add (converter calculators)[https://github.com/openvinotoolkit/mediapipe/tree/main/mediapipe/calculators/openvino] at the beggining in end. You may also need to write your own conversion calculator.

### 5. Fifth step may be required if you plan to use existing graph within existing application but use OVMS for inference
For build instructions check documentation [here](TODO). Additionally you need to fill `server_config` field in `OpenVINOModelServerSessionCalculatorOptions` to pass full file path to configuration file like:
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

