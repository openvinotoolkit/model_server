# Efficient Image Generation Serving {#ovms_docs_image_generation_reference}

## Image Generation Calculator
Image Generation pipeline consists of one MediaPipe node - Image Generation Calculator. To serve the image generation model, it is required to create a MediaPipe graph configuration file that defines the node and its parameters. The graph configuration file is typically named `graph.pbtxt` and is placed in the model directory.
The `graph.pbtxt` file may be created automatically by the Model Server when [using HuggingFaces pulling](../pull_hf_models.md) on start-up, automatically via [export models script](../../demos/common/export_models/) or manually by an administrator.

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

The input JSON content should be compatible with the [Image Generation API](../model_server_rest_api_image_generation.md).

The input also includes a side packet with a reference to `IMAGE_GEN_NODE_RESOURCES` which is a shared object representing multiple OpenVINO GenAI pipelines built from OpenVINO models loaded into memory just once.

**Every node based on Image Generation Calculator MUST have exactly that specification of this side packet:**

`input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"`

**If it is missing or modified, model server will fail to provide graph with the model**

The calculator produces `std::string` MediaPipe packet with the JSON content representing OpenAI response format, [described in separate document](../model_server_rest_api_image_generation.md). Image Generation calculator has no support for streaming and partial responses.

Let's have a look at the example graph definition:
```protobuf
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "./"
          device: "CPU"
      }
  }
}
```

Above node configuration should be used as a template since user is not expected to change most of it's content. Actually only `node_options` requires user attention as it specifies OpenVINO GenAI pipeline parameters. The rest of the configuration can remain unchanged.

The calculator supports the following `node_options` for tuning the pipeline configuration:
-    `required string models_path` - location of the models and scheduler directory (can be relative);
-    `optional string device` - device to load models to. Supported values: "CPU", "GPU", "NPU" [default = "CPU"]
-    `optional string plugin_config` - [OpenVINO device plugin configuration](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes.html) and additional pipeline options. Should be provided in the same format for regular [models configuration](../parameters.md#model-configuration-options). The config is used for all models in the pipeline except for tokenizers (text encoders/decoders, unet, vae) [default = "{}"]
-    `optional string max_resolution` - maximum resolution allowed for generation. Requests exceeding this value will be rejected. [default = "4096x4096"];
-    `optional string default_resolution` - default resolution used for generation. If not specified, underlying model shape will determine final resolution.
-    `optional uint64 max_num_images_per_prompt` - maximum number of images generated per prompt. Requests exceeding this value will be rejected. [default = 10];
-    `optional uint64 default_num_inference_steps` - default number of inference steps used for generation, if not specified by the request [default = 50];
-    `optional uint64 max_num_inference_steps` - maximum number of inference steps allowed for generation. Requests exceeding this value will be rejected. [default = 100];


## Models Directory

In node configuration we set `models_path` indicating location of the directory with files loaded by LLM engine. It loads following files:

```
models/OpenVINO/
├── FLUX.1-schnell-int4-ov
│   ├── graph.pbtxt <----------------- - OVMS MediaPipe graph configuration file
│   ├── model_index.json <------------ - GenAI configuration file including pipeline type SD/SDXL/SD3/FLUX
│   ├── README.md
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── text_encoder
│   │   ├── config.json
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   ├── text_encoder_2
│   │   ├── config.json
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── openvino_detokenizer.bin
│   │   ├── openvino_detokenizer.xml
│   │   ├── openvino_tokenizer.bin
│   │   ├── openvino_tokenizer.xml
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── tokenizer_2
│   │   ├── openvino_detokenizer.bin
│   │   ├── openvino_detokenizer.xml
│   │   ├── openvino_tokenizer.bin
│   │   ├── openvino_tokenizer.xml
│   │   ├── special_tokens_map.json
│   │   ├── spiece.model
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── transformer
│   │   ├── config.json
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   ├── vae_decoder
│   │   ├── config.json
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   └── vae_encoder
│       ├── config.json
│       ├── openvino_model.bin
│       └── openvino_model.xml
└── stable-diffusion-v1-5-int8-ov
    ├── feature_extractor
    │   └── preprocessor_config.json
    ├── graph.pbtxt <----------------- - OVMS MediaPipe graph configuration file
    ├── model_index.json <------------ - GenAI configuration file including pipeline type SD/SDXL/SD3/FLUX
    ├── README.md
    ├── safety_checker
    │   ├── config.json
    │   └── model.safetensors
    ├── scheduler
    │   └── scheduler_config.json
    ├── text_encoder
    │   ├── config.json
    │   ├── openvino_model.bin
    │   └── openvino_model.xml
    ├── tokenizer
    │   ├── merges.txt
    │   ├── openvino_detokenizer.bin
    │   ├── openvino_detokenizer.xml
    │   ├── openvino_tokenizer.bin
    │   ├── openvino_tokenizer.xml
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.json
    ├── unet
    │   ├── config.json
    │   ├── openvino_model.bin
    │   └── openvino_model.xml
    ├── vae_decoder
    │   ├── config.json
    │   ├── openvino_model.bin
    │   └── openvino_model.xml
    └── vae_encoder
        ├── config.json
        ├── openvino_model.bin
        └── openvino_model.xml

```

- `graph.pbtxt` - MediaPipe graph configuration file defining the Image Generation Calculator node and its parameters.
- `model_index.json` - GenAI configuration file that describes the pipeline type (SD/SDXL/SD3/FLUX) and the models used in the pipeline.
- `scheduler/scheduler_config.json` - configuration file for the scheduler that manages the execution of the models in the pipeline.
- `text_encoder`, `tokenizer`, `unet`, `vae_encoder`, `vae_decoder` - directories containing the OpenVINO models and their configurations for the respective components of the image generation pipeline.

We recommend using [export script](../../demos/common/export_models/README.md) to prepare models directory structure for serving, or simply use [HuggingFace pulling](../pull_hf_models.md) to automatically download and convert models from Hugging Face Hub.

Check [tested models](https://github.com/openvinotoolkit/openvino.genai/blob/master/tests/python_tests/models/real_models).

## References
- [Image Generation API](../model_server_rest_api_image_generation.md)
- Demos on [CPU/GPU](../../demos/image_generation/README.md)
