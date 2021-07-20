# Overview of OpenVINO&trade; Model Server Resources


## Samples

These samples include python scripts and installation steps to use OpenVINO&trade; Model Server for various use-cases. 

- [Face Detection with OpenVINO&trade; Model Server](./face_detection_script_example.md#example-of-face-detection-with-openvino-model-server) : This sample uses python script to run OpenVINO&trade; Face Detection Model with the Model Server.
- [ONNX model serving](ovms_onnx_example.md)
- [Person-Vehicle-Bike Detection](./face_detection_script_example.md#running-person-vehicle-detection-with-example-script) : Detection of Person and Vehicles using OpenVINO&trade; Model Server with python example.
- [Age Gender Classification](./age_gender_guide.md) : Classification of Age Gender using OpenVINO&trade; Model Server.
- [Processing frames from camera in parallel](./camera_example.md) : Horizontal text detection in real-time
- [Binary Input](./binary_input.md) Sending prediction requests with images in JPEG or PNG format.
- [DAG Scheduler pipeline with models ensemble](ensemble_scheduler.md)
- [DAG Scheduler with OCR pipeline](east_ocr.md)
- [DAG Scheduler pipeline with combined models](combined_model_dag.md)
- [DAG Scheduler with vehicle detection](vehicles_analysis_dag.md)
- [Dynamic batch size with demultiplexer](dynamic_batch_size.md)
- [Stateful model for audio recognition](stateful_models.md)     
- [Clients for gRPC and REST API calls](https://github.com/openvinotoolkit/model_server/tree/main/example_client) : Example python scripts for gRPC and REST API calls with resnet classification models

## Additional Tools

- [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html) : Convert TensorFLow, Caffe, etc models to IR models

## Pre-trained Models

- [Intel's Pre-trained Models from Open Model Zoo](https://docs.openvinotoolkit.org/latest/omz_models_intel_index.html)
- [Public Pre-trained Models Available with OpenVINO&trade; from Open Model Zoo](https://docs.openvinotoolkit.org/latest/omz_models_public_index.html)