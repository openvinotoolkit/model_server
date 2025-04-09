# Demos {#ovms_docs_demos}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_demos_rerank
ovms_demos_embeddings
ovms_demos_continuous_batching
ovms_demos_continuous_batching_vlm
ovms_demos_llm_npu
ovms_demos_vlm_npu
ovms_demo_clip_image_classification
ovms_demo_age_gender_guide
ovms_demo_horizontal_text_detection
ovms_demo_optical_character_recognition
ovms_demo_face_detection
ovms_demo_face_blur_pipeline
ovms_demo_capi_inference_demo
ovms_demo_single_face_analysis_pipeline
ovms_demo_multi_faces_analysis_pipeline
ovms_docs_demo_ensemble
ovms_docs_demo_mediapipe_image_classification
ovms_docs_demo_mediapipe_multi_model
ovms_docs_demo_mediapipe_object_detection
ovms_docs_demo_mediapipe_holistic
ovms_docs_demo_mediapipe_iris
ovms_docs_image_classification
ovms_demo_using_onnx_model
ovms_demo_tf_classification
ovms_demo_person_vehicle_bike_detection
ovms_demo_vehicle_analysis_pipeline
ovms_demo_real_time_stream_analysis
ovms_demo_using_paddlepaddle_model
ovms_demo_bert
ovms_demo_universal-sentence-encoder
ovms_demo_benchmark_client
ovms_demo_python_seq2seq
ovms_demo_python_stable_diffusion
ovms_string_output_model_demo

```

OpenVINO Model Server demos have been created to showcase the usage of the model server as well as demonstrate it’s capabilities.
### Check Out New Generative AI Demos
| Demo | Description |
|---|---|
|[LLM Text Generation with continuous batching](continuous_batching/README.md)|Generate text with LLM models and continuous batching pipeline|
|[VLM Text Generation with continuous batching](continuous_batching/vlm/README.md)|Generate text with VLM models and continuous batching pipeline|
|[OpenAI API text embeddings ](embeddings/README.md)|Get text embeddings via endpoint compatible with OpenAI API|
|[Reranking with Cohere API](rerank/README.md)| Rerank documents via endpoint compatible with Cohere|
|[RAG with OpenAI API endpoint and langchain](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb)| Example how to use RAG with model server endpoints|
|[LLM on NPU](./llm_npu/README.md)| Generate text with LLM models and NPU acceleration|
|[VLM on NPU](./vlm_npu/README.md)| Generate text with VLM models and NPU acceleration|
|[VisualCode assistant](./code_completion_copilot/README.md)|Use Continue extension in Visual Studio Code with local OVMS|


Check out the list below to see complete step-by-step examples of using OpenVINO Model Server with real world use cases:

## With Traditional Models
| Demo | Description |
|---|---|
|[Image Classification](image_classification/python/README.md)|Run prediction on a JPEG image using image classification model via gRPC API.|
|[Using ONNX Model](using_onnx_model/python/README.md)|Run prediction on a JPEG image using image classification ONNX model via gRPC API in two preprocessing variants. This demo uses [pipeline](../docs/dag_scheduler.md) with [image_transformation custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2025/1/src/custom_nodes/image_transformation). |
|[Using TensorFlow Model](image_classification_using_tf_model/python/README.md)|Run image classification using directly imported TensorFlow model. |
|[Age gender recognition](age_gender_recognition/python/README.md) | Run prediction on a JPEG image using age gender recognition model via gRPC API.|
|[Face Detection](face_detection/python/README.md)|Run prediction on a JPEG image using face detection model via gRPC API.|
|[Classification with PaddlePaddle](classification_using_paddlepaddle_model/python/README.md)| Perform classification on an image with a PaddlePaddle model. |
|[Natural Language Processing with BERT](bert_question_answering/python/README.md)|Provide a knowledge source and a query and use BERT model for question answering use case via gRPC API. This demo uses dynamic shape feature. |
|[Using inputs data in string format with universal-sentence-encoder model](universal-sentence-encoder/README.md)| Handling AI model with text as the model input. |
|[Person, Vehicle, Bike Detection](person_vehicle_bike_detection/python/README.md)|Run prediction on a video file or camera stream using person, vehicle, bike detection model via gRPC API.|
|[Benchmark App](benchmark/python/README.md)|Generate traffic and measure performance of the model served in OpenVINO Model Server.|

## With Python Nodes
| Demo | Description |
|---|---|
|[Stable Diffusion](python_demos/stable_diffusion/README.md) | Generate image using Stable Diffusion model sending prompts via gRPC API unary or interactive streaming endpoint.|
|[CLIP image classification](python_demos/clip_image_classification/README.md) | Classify image according to provided labels using CLIP model embedded in a multi-node MediaPipe graph.|
|[Seq2seq translation](python_demos/seq2seq_translation/README.md) | Translate text using seq2seq model via gRPC API.|

## With MediaPipe Graphs
| Demo | Description |
|---|---|
|[Real Time Stream Analysis](real_time_stream_analysis/python/README.md)| Analyze RTSP video stream in real time with generic application template for custom pre and post processing routines as well as simple results visualizer for displaying predictions in the browser. |
|[Image classification](./mediapipe/image_classification/README.md)| Basic example with a single inference node. |
|[Chain of models](./mediapipe/image_classification/README.md)| A chain of models in a graph. |
|[Object detection](./mediapipe/object_detection/README.md)| A pipeline implementing object detection |
|[Iris demo](./mediapipe/object_detection/README.md)| A pipeline implementing iris detection |
|[Holistic demo](./mediapipe/holistic_tracking/README.md)| A complex pipeline linking several image analytical models and image transformations |

## With DAG Pipelines
| Demo | Description |
|---|---|
|[Horizontal Text Detection in Real-Time](horizontal_text_detection/python/README.md) | Run prediction on camera stream using a horizontal text detection model via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [horizontal_ocr custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2025/1/src/custom_nodes/horizontal_ocr) and [demultiplexer](../docs/demultiplexing.md). |
|[Optical Character Recognition Pipeline](optical_character_recognition/python/README.md) | Run prediction on a JPEG image using a pipeline of text recognition and text detection models with a custom node for intermediate results processing via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [east_ocr custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2025/1/src/custom_nodes/east_ocr) and [demultiplexer](../docs/demultiplexing.md). |
|[Single Face Analysis Pipeline](single_face_analysis_pipeline/python/README.md)|Run prediction on a JPEG image using a simple pipeline of age-gender recognition and emotion recognition models via gRPC API to analyze image with a single face. This demo uses [pipeline](../docs/dag_scheduler.md) |
|[Multi Faces Analysis Pipeline](multi_faces_analysis_pipeline/python/README.md)|Run prediction on a JPEG image using a pipeline of age-gender recognition and emotion recognition models via gRPC API to extract multiple faces from the image and analyze all of them. This demo uses [pipeline](../docs/dag_scheduler.md) with [model_zoo_intel_object_detection custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2025/1/src/custom_nodes/model_zoo_intel_object_detection) and [demultiplexer](../docs/demultiplexing.md) |
|[Model Ensemble Pipeline](model_ensemble/python/README.md)|Combine multiple image classification models into one [pipeline](../docs/dag_scheduler.md) and aggregate results to improve classification accuracy. |
|[Face Blur Pipeline](face_blur/python/README.md)|Detect faces and blur image using a pipeline of object detection models with a custom node for intermediate results processing via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [face_blur custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2025/1/src/custom_nodes/face_blur). |
|[Vehicle Analysis Pipeline](vehicle_analysis_pipeline/python/README.md)|Detect vehicles and recognize their attributes using a pipeline of vehicle detection and vehicle attributes recognition models with a custom node for intermediate results processing via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [model_zoo_intel_object_detection custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2025/1/src/custom_nodes/model_zoo_intel_object_detection). |

## With C++ Client
| Demo | Description |
|---|---|
|[C API applications](c_api_minimal_app/README.md)|How to use C API from the OpenVINO Model Server to create C and C++ application.|

## With Go Client
| Demo | Description |
|---|---|
|[Image Classification](image_classification/go/README.md)|Run prediction on a JPEG image using image classification model via gRPC API.|


