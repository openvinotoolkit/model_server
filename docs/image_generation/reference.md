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

The input multi-part content should be compatible with the [Image Edit API](../model_server_rest_api_image_edit.md).

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
-    `optional string device` - device to load models to. Supported values: "CPU", "GPU", "NPU" [default = "CPU"] or mixed - space separated. Example: `CPU GPU NPU` equals to `text_encode=CPU denoise=GPU vae=NPU`
-    `optional string plugin_config` - [OpenVINO device plugin configuration](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes.html) and additional pipeline options. Should be provided in the same format for regular [models configuration](../parameters.md#model-configuration-options). The config is used for all models in the pipeline except for tokenizers (text encoders/decoders, unet, vae) [default = "{}"]
-    `optional string max_resolution` - maximum resolution allowed for generation. Requests exceeding this value will be rejected. [default = "4096x4096"];
-    `optional string default_resolution` - default resolution used for generation. If not specified, underlying model shape will determine final resolution.
-    `optional uint64 max_num_images_per_prompt` - maximum number of images generated per prompt. Requests exceeding this value will be rejected. [default = 10];
-    `optional uint64 default_num_inference_steps` - default number of inference steps used for generation, if not specified by the request [default = 50];
-    `optional uint64 max_num_inference_steps` - maximum number of inference steps allowed for generation. Requests exceeding this value will be rejected. [default = 100];

Static model resolution settings:
-    `optional string resolution` - enforces static resolution for all requests. When specified, underlying models are reshaped to this resolution.
-    `optional uint64 num_images_per_prompt` - used together with max_resolution, to define batch size in static model shape.
-    `optional float guidance_scale` - used together with max_resolution

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

## LoRA Adapters

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) adapters allow fine-tuning image generation models without retraining the full model. OVMS supports loading multiple LoRA adapters at startup and dynamically selecting/blending them per request.

### Registering LoRA Adapters

LoRA adapters are registered at server startup via the `--source_loras` CLI parameter. The format is a comma-separated list of `alias=source` entries:

```
--source_loras=alias1=source1,alias2=source2,...
```

**Supported source types:**

| Source Type | Format | Example |
|------------|--------|---------|
| HuggingFace repo | `org/repo` | `pokemon=juliensimon/sd-pokemon-lora` |
| HuggingFace repo with explicit file | `org/repo@filename.safetensors` | `xray=DoctorDiffusion/doctor-diffusion-s-xray-xl-lora@DD-xray-v1.safetensors` |
| Direct URL | `https://...` | `style=https://huggingface.co/user/repo/resolve/main/model.safetensors` |
| Local file path | `/path/to/file.safetensors` | `custom=/models/loras/my_style.safetensors` |

**Example:**
```bash
ovms --rest_port 8000 \
  --model_repository_path /models/ \
  --task image_generation \
  --source_model stabilityai/stable-diffusion-xl-base-1.0 \
  --source_loras "xray=DoctorDiffusion/doctor-diffusion-s-xray-xl-lora@DD-xray-v1.safetensors,ukiyo=KappaNeuro/ukiyo-e-art@Ukiyo-e Art.safetensors,vector=DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora@DD-vector-v2.safetensors"
```

> **Important:** LoRA adapters must be compatible with the base model architecture. For example, SDXL adapters can only be used with an SDXL base model.

### Composite Adapters

You can define composite adapters that blend multiple adapters with specified weights:

```
--source_loras="pokemon=juliensimon/sd-pokemon-lora,anime=user/anime-lora,mix=@pokemon:0.7+@anime:0.5"
```

The `mix` adapter is a composite that blends `pokemon` at weight 0.7 and `anime` at weight 0.5.

### Per-Request LoRA Selection via Model Name Routing

Adapter selection is driven by the `model` field in the request. When the `model` field matches a registered adapter alias, that adapter is automatically applied:

```bash
curl http://localhost:8000/v3/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "xray", "prompt": "xray a human hand", "num_inference_steps": 20}'
```

In this example, `xray` is the alias defined in `--source_loras` (e.g. `xray=DoctorDiffusion/doctor-diffusion-s-xray-xl-lora@DD-xray-v1.safetensors`). The adapter is applied with its default weight.

When the `model` field matches a **composite** adapter alias, all component adapters are activated with their pre-defined weights:

```bash
curl http://localhost:8000/v3/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "mix", "prompt": "a landscape"}'
```

When the `model` field is the **base model name** (not matching any adapter alias), generation proceeds without any LoRA adapter applied (base model only).

### Overriding Adapter Weights with `lora_weights`

The `lora_weights` field in the request body allows overriding the default weight of the active adapter(s). It does **not** independently select which adapters to activate — adapter selection is always based on the `model` field.

**Override a single adapter weight:**
```json
{
  "model": "xray",
  "prompt": "xray a cute cat in sunglasses",
  "lora_weights": {"xray": 0.5},
  "num_inference_steps": 20
}
```

**Override component weights in a composite adapter:**
```json
{
  "model": "mix",
  "prompt": "a landscape in mixed style",
  "lora_weights": {"ukiyo": 0.3, "vector": 0.8}
}
```

### Blending Multiple Adapters

To blend multiple adapters simultaneously, define a **composite adapter** at startup:

```
--source_loras="xray=DoctorDiffusion/doctor-diffusion-s-xray-xl-lora@DD-xray-v1.safetensors,ukiyo=KappaNeuro/ukiyo-e-art@Ukiyo-e Art.safetensors,blend=@xray:0.5+@ukiyo:0.4"
```

Then use the composite alias in requests:
```bash
curl http://localhost:8000/v3/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "blend", "prompt": "a cat"}'
```

You can override individual component weights at request time via `lora_weights`:
```json
{
  "model": "blend",
  "prompt": "a cat",
  "lora_weights": {"xray": 0.8, "ukiyo": 0.2}
}
```

## References
- [Image Generation API](../model_server_rest_api_image_generation.md)
- [Image Edit API](../model_server_rest_api_image_edit.md)
- Demos on [CPU/GPU](../../demos/image_generation/README.md)
