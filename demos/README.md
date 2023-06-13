# Demos {#ovms_docs_demos}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

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
   ovms_docs_image_classification
   ovms_demo_using_onnx_model
   ovms_demo_tf_classification
   ovms_demo_person_vehicle_bike_detection
   ovms_demo_vehicle_analysis_pipeline
   ovms_demo_real_time_stream_analysis
   ovms_demo_bert
   ovms_demo_gptj_causal_lm
   ovms_demo_universal-sentence-encoder
   ovms_demo_speech_recognition
   ovms_demo_benchmark_client

@endsphinxdirective

OpenVINO Model Server demos have been created to showcase the usage of the model server as well as demonstrate it’s capabilities. Check out the list below to see complete step-by-step examples of using OpenVINO Model Server with real world use cases:

## Python 
| Demo | Description |
|---|---|
|[Age gender recognition](age_gender_recognition/python/README.md) | Run prediction on a JPEG image using age gender recognition model via gRPC API.|
|[Horizontal Text Detection in Real-Time](horizontal_text_detection/python/README.md) | Run prediction on camera stream using a horizontal text detection model via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [horizontal_ocr custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/horizontal_ocr) and [demultiplexer](../docs/demultiplexing.md). |
|[Optical Character Recognition Pipeline](optical_character_recognition/python/README.md) | Run prediction on a JPEG image using a pipeline of text recognition and text detection models with a custom node for intermediate results processing via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [east_ocr custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/east_ocr) and [demultiplexer](../docs/demultiplexing.md). |
|[Face Detection](face_detection/python/README.md)|Run prediction on a JPEG image using face detection model via gRPC API.|
|[Single Face Analysis Pipeline](single_face_analysis_pipeline/python/README.md)|Run prediction on a JPEG image using a simple pipeline of age-gender recognition and emotion recognition models via gRPC API to analyze image with a single face. This demo uses [pipeline](../docs/dag_scheduler.md) |
|[Multi Faces Analysis Pipeline](multi_faces_analysis_pipeline/python/README.md)|Run prediction on a JPEG image using a pipeline of age-gender recognition and emotion recognition models via gRPC API to extract multiple faces from the image and analyze all of them. This demo uses [pipeline](../docs/dag_scheduler.md) with [model_zoo_intel_object_detection custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/model_zoo_intel_object_detection) and [demultiplexer](../docs/demultiplexing.md) |
|[Model Ensemble Pipeline](model_ensemble/python/README.md)|Combine multiple image classification models into one [pipeline](../docs/dag_scheduler.md) and aggregate results to improve classification accuracy. |
|[Image Classification](image_classification/python/README.md)|Run prediction on a JPEG image using image classification model via gRPC API.|
|[Using ONNX Model](using_onnx_model/python/README.md)|Run prediction on a JPEG image using image classification ONNX model via gRPC API in two preprocessing variants. This demo uses [pipeline](../docs/dag_scheduler.md) with [image_transformation custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/image_transformation). |
|[Using TensorFlow Model](image_classification_using_tf_model/python/README.md)|Run image classification using directly imported TensorFlow model. |
|[Person, Vehicle, Bike Detection](person_vehicle_bike_detection/python/README.md)|Run prediction on a video file or camera stream using person, vehicle, bike detection model via gRPC API.|
|[Vehicle Analysis Pipeline](vehicle_analysis_pipeline/python/README.md)|Detect vehicles and recognize their attributes using a pipeline of vehicle detection and vehicle attributes recognition models with a custom node for intermediate results processing via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [model_zoo_intel_object_detection custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/model_zoo_intel_object_detection). |
|[Real Time Stream Analysis](real_time_stream_analysis/python/README.md)| Analyze RTSP video stream in real time with generic application template for custom pre and post processing routines as well as simple results visualizer for displaying predictions in the browser. |
|[Natural Language Processing with BERT](bert_question_answering/python/README.md)|Provide a knowledge source and a query and use BERT model for question answering use case via gRPC API. This demo uses dynamic shape feature. |
|[GPT-J Causal Language Modeling](gptj_causal_lm/python/README.md)|Write start of the sentence and let GPT-J continue via gRPC API. This demo uses dynamic shape feature. |
|[Using inputs data in string format with universal-sentence-encoder model](universal-sentence-encoder/README.md)| Handling AI model with text as the model input. | 
|[Speech Recognition on Kaldi Model](speech_recognition_with_kaldi_model/python/README.md)|Run inference on a speech sample and use Kaldi model to perform speech recognition via gRPC API. This demo uses [stateful model](../docs/stateful_models.md). |
|[Benchmark App](benchmark/python/README.md)|Generate traffic and measure performance of the model served in OpenVINO Model Server.|
|[Face Blur Pipeline](face_blur/python/README.md)|Detect faces and blur image using a pipeline of object detection models with a custom node for intermediate results processing via gRPC API. This demo uses [pipeline](../docs/dag_scheduler.md) with [face_blur custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/face_blur). |

## C++
| Demo | Description |
|---|---|
|[C API applications](c_api_minimal_app/README.md)|How to use C API from the OpenVINO Model Server to create C and C++ application.|
|[Image Classification](image_classification/cpp/README.md)|Run prediction on a JPEG image using image classification model via gRPC API.|
|[Benchmark App](benchmark/cpp/README.md)|Generate traffic and measure performance of the model served in OpenVINO Model Server.|

## Go
| Demo | Description |
|---|---|
|[Image Classification](image_classification/go/README.md)|Run prediction on a JPEG image using image classification model via gRPC API.|
