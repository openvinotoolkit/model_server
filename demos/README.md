# OpenVINO™ Model Server Demos

OpenVINO Model Server demos have been created to showcase the usage of the model server as well as demonstrate it’s capabilities. Check out the list below to see complete step-by-step examples of using OpenVINO Model Server with real world use cases:

## Python 
| Demo | Description |
|---|---|
|[Horizontal Text Detection in Real-Time](horizontal_text_detection/python) | Run prediction on camera stream using a horizontal text detection model via gRPC API._This demo uses [pipeline](../docs/dag_scheduler.md) with [horizontal_ocr custom node](../src/custom_nodes/horizontal_ocr) and [demultiplexer](../docs/demultiplexing.md)_|
|[Optical Character Recognition Pipeline](optical_character_recognition/python) | Run prediction on a JPEG image using a pipeline of text recognition and text detection models with a custom node for intermediate results processing via gRPC API. _This demo uses [pipeline](../docs/dag_scheduler.md) with [east_ocr custom node](../src/custom_nodes/east_ocr) and [demultiplexer](../docs/demultiplexing.md)_|
|[Face Detection](face_detection/python)|Run prediction on a JPEG image using face detection model via gRPC API.|
|[Face Analysis Pipeline](face_anaysis_pipeline/python)|Run prediction on a JPEG image using a pipeline of age-gender recognition and emotion recogition models via gRPC API._This demo uses [pipeline](../docs/dag_scheduler.md) with [model_zoo_intel_object_detection custom node](../src/custom_nodes/model_zoo_intel_object_detection) and [demultiplexer](../docs/demultiplexing.md)_|
|[Image Classification](image_classification/python)|Run prediction on a JPEG image using image classification model via gRPC API.|
|[Person, Vehicle, Bike Detection](person_bike_vehicle_detection/python)|Run prediction on a video file or camera stream using person, vehicle, bike detection model via gRPC API.|
|[Vehicle Analysis Pipeline](vehicle_analysis_pipeline/python)|Detect vehicles and recognize their attributes using a pipeline of vehicle detection and vehicle attributes recognition models with a custom node for intermediate results processing via gRPC API. _This demo uses [pipeline](../docs/dag_scheduler.md) with [model_zoo_intel_object_detection custom node](../src/custom_nodes/model_zoo_intel_object_detection)_||
|[Image Transformation Node](image_transformation_node/python)|Learn about image transformation custom node that can transform input image to the format expected by the deep learning model. _This demo uses [image_transformation custom node](../src/custom_nodes/image_transformation)_|
|[Natural Language Processing with BERT](bert_question_answering/python)|Provide a knowledge source and a query and use BERT model for question answering use case via gRPC API.|
|[Speech Recognition on Kaldi Model](speech_recognition_with_kaldi_model/python)|Run inference on a speech sample and use Kaldi model to perform speech recognition via gRPC API. _This demo uses [stateful model](../docs/stateful_models)_|
|[Benchmark App](benchmark/python)|Generate traffic and measure performance of the model served in OpenVINO Model Server.|

## C++
| Demo | Description |
|---|---|
|[Image Classification](image_classification/cpp)|Run prediction on a JPEG image using image classification model via gRPC API.|
|[Benchmark App](benchmark/cpp)|Generate traffic and measure performance of the model served in OpenVINO Model Server.|

## Go
| Demo | Description |
|---|---|
|[Image Classification](image_classification/go)|Run prediction on a JPEG image using image classification model via gRPC API.|