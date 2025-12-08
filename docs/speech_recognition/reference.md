# Speech to text models Serving {#ovms_docs_speech_to_text_reference}

## Speech to text Calculator
Speech to text pipeline consists of one MediaPipe node - S2T Calculator. To serve the speech generation model, it is required to create a MediaPipe graph configuration file that defines the node and its parameters. The graph configuration file is typically named `graph.pbtxt` and is placed in the model directory.
The `graph.pbtxt` file may be created automatically via [export models script](../../demos/common/export_models/) or manually by an administrator.

Calculator has access to HTTP request and parses it to extract the generation parameters:
```cpp
struct HttpPayload {
    std::string uri;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::shared_ptr<rapidjson::Document> parsedJson;
    std::shared_ptr<ClientConnection> client;
    std::shared_ptr<MultiPartParser> multipartParser;
};
```

The input JSON content should be compatible with the [Transcription API](../model_server_rest_api_speech_to_text.md#transcription) or [Translation API](../model_server_rest_api_speech_to_text.md#translation).

The input also includes a side packet with a reference to `STT_NODE_RESOURCES` which is a shared object representing multiple OpenVINO GenAI pipelines built from OpenVINO models loaded into memory just once.

**Every node based on Speech Generation Calculator MUST have exactly that specification of this side packet:**

`input_side_packet: "STT_NODE_RESOURCES:s2t_servable"`

**If it is missing or modified, model server will fail to provide graph with the model**

The calculator produces `std::string` MediaPipe packet with the wave audio file content. Speech Generation calculator has no support for streaming and partial responses.

Let's have a look at the example graph definition:
```protobuf
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node {
  name: "S2tExecutor"
  input_side_packet: "STT_NODE_RESOURCES:s2t_servable"
  calculator: "S2tCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
    [type.googleapis.com / mediapipe.S2tCalculatorOptions]: {
      models_path: "./",
      target_device: "CPU"
    }
  }
}
```

Above node configuration should be used as a template since user is not expected to change most of it's content. Actually only `node_options` requires user attention as it specifies OpenVINO GenAI pipeline parameters. The rest of the configuration can remain unchanged.

The calculator supports the following `node_options` for tuning the pipeline configuration:
-    `required string models_path` - location of the models and scheduler directory (can be relative);
-    `optional string device` - device to load models to. Supported values: "CPU, GPU, NPU" [default = "CPU"]

We recommend using [export script](../../demos/common/export_models/README.md) to prepare models directory structure for serving.
Check [supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models).

### Text to speech calculator limitations
- Streaming is not supported

## References
- [Transcription API](../model_server_rest_api_speech_to_text.md#transcription)
- [Translation API](../model_server_rest_api_speech_to_text.md#translation)
- Demos on [CPU/GPU](../../demos/audio/README.md#transcription)
