# Demos {#ovms_docs_demos}

```{toctree}
---
maxdepth: 1
hidden:
---

Text generation <ovms_text_generation>
Image generation <ovms_demos_image_generation>
Audio <ovms_demos_audio>
Text Embeddings <ovms_demos_embeddings>
Text Reranking <ovms_demos_rerank>
Classic models <ovms_demos_classic_models>
MediaPipe <ovms_demos_mediapipe>
Python Node <ovms_demos_python_node>
Integrations <ovms_demos_integrations>
```

## Text Generation
| Demo | Description |
|---|---|
|[LLM Text Generation with continuous batching](continuous_batching/README.md)|Generate text with LLM models and continuous batching pipeline.|
|[VLM Text Generation with continuous batching](continuous_batching/vlm/README.md)|Generate text with VLM models and continuous batching pipeline.|
|[AI Agents with MCP servers](./continuous_batching/agentic_ai/README.md)|OpenAI agents with MCP servers and serving LLM models.|
|[RAG with OpenAI API endpoint and langchain](continuous_batching/rag/README.md)|Example how to use RAG with model server endpoints.|
|[Long context LLMs](./continuous_batching/long_context/README.md)|Recommendations for handling very long context in LLM models.|
|[Structured output](./continuous_batching/structured_output/README.md)|Generate structured (JSON) output from LLM models.|
|[Speculative decoding](./continuous_batching/speculative_decoding/README.md)|Speed up LLM inference with speculative decoding.|
|[LLM on NPU](./llm_npu/README.md)|Generate text with LLM models and NPU acceleration.|
|[Scaling on multi CPU and GPU](./continuous_batching/scaling/README.md)|Scale LLM serving across multiple CPUs and GPUs.|
|[Loading models in GGUF](gguf/README.md)|Serve GGUF models with OVMS.|


## Image Generation
| Demo | Description |
|---|---|
|[Image Generation](image_generation/README.md)|Generate images with diffusion models.|

## Audio
| Demo | Description |
|---|---|
|[Audio demos](audio/README.md)|Text-to-speech and automatic speech recognition demos.|

## Text Embeddings
| Demo | Description |
|---|---|
|[OpenAI API text embeddings](embeddings/README.md)|Get text embeddings via endpoint compatible with OpenAI API.|

## Text Reranking
| Demo | Description |
|---|---|
|[Reranking with Cohere API](rerank/README.md)|Rerank documents via endpoint compatible with Cohere.|

## Classic Models
| Demo | Description |
|---|---|
|[Image Classification](image_classification/python/README.md)|Run prediction on a JPEG image using image classification model via gRPC API.|
|[Using ONNX Model](using_onnx_model/python/README.md)|Run prediction on a JPEG image using image classification ONNX model via gRPC API in two preprocessing variants. This demo uses [pipeline](../docs/dag_scheduler.md) with [image_transformation custom node](https://github.com/openvinotoolkit/model_server/tree/main/src/custom_nodes/image_transformation).|
|[Using TensorFlow Model](image_classification_using_tf_model/python/README.md)|Run image classification using directly imported TensorFlow model.|
|[Classification with PaddlePaddle](classification_using_paddlepaddle_model/python/README.md)|Perform classification on an image with a PaddlePaddle model.|
|[Age gender recognition](age_gender_recognition/python/README.md)|Run prediction on a JPEG image using age gender recognition model via gRPC API.|
|[Face Detection](face_detection/python/README.md)|Run prediction on a JPEG image using face detection model via gRPC API.|
|[Person, Vehicle, Bike Detection](person_vehicle_bike_detection/python/README.md)|Run prediction on a video file or camera stream using person, vehicle, bike detection model via gRPC API.|
|[Using input strings](universal-sentence-encoder/README.md)|Handling AI model with text as the model input.|
|[Using output strings](image_classification_with_string_output/README.md)|Handling AI model with string output.|
|[Natural Language Processing with BERT](bert_question_answering/python/README.md)|Provide a knowledge source and a query and use BERT model for question answering via gRPC API. This demo uses dynamic shape feature.|
|[Benchmark App](benchmark/python/README.md)|Generate traffic and measure performance of the model served in OpenVINO Model Server.|

## MediaPipe
| Demo | Description |
|---|---|
|[Object Detection](./mediapipe/object_detection/README.md)|A pipeline implementing object detection.|
|[Iris](./mediapipe/iris_tracking/README.md)|A pipeline implementing iris detection.|
|[Holistic](./mediapipe/holistic_tracking/README.md)|A complex pipeline linking several image analytical models and image transformations.|
|[Realtime Stream Analysis](real_time_stream_analysis/python/README.md)|Analyze RTSP video stream in real time with generic application template for custom pre and post processing routines.|
|[Image classification](./mediapipe/image_classification/README.md)|Basic example with a single inference node.|
|[Chain of models](./mediapipe/multi_model_graph/README.md)|A chain of models in a graph.|
|[CLIP image classification](python_demos/clip_image_classification/README.md)|Classify image according to provided labels using CLIP model embedded in a multi-node MediaPipe graph.|

## Python Node
| Demo | Description |
|---|---|
|[OpenClip with python execution](./python_demos/clip_image_classification/README.md)|A pipeline implementing OpenClip classification in Python Node.|

## Integrations
| Demo | Description |
|---|---|
|[Integration with Open WebUI](integration_with_OpenWebUI/README.md)|Using Open WebUI with OVMS as inference provider. Shows text and image generation as well as usage with RAG and tools.|
|[Visual Studio Code assistant](./code_local_assistant/README.md)|Use Continue or Cline extension to Visual Studio Code with local OVMS serving.|
