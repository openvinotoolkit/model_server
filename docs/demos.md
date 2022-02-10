# OpenVINO Model Server Demos {#ovms_docs_demos}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_demo_age_gender_guide
   ovms_docs_demo_camera_example
   ovms_docs_demo_combined_model_dag
   ovms_docs_demo_dynamic_batch_demuliplexer
   ovms_docs_demo_ocr
   ovms_docs_demo_ensemble
   ovms_docs_demo_face_detection
   ovms_docs_demo_face_analysis_dag
   ovms_docs_demo_onnx
   ovms_docs_demo_tensorflow_conversion
   ovms_docs_demo_vehicle_analysis
   ovms_example_client_bert_readme
   ovms_example_client_cpp_readme
   ovms_example_client_go_readme
   ovms_client_python_samples_readme

@endsphinxdirective

OpenVINO Model Server demos show how to use the model server and its features. Source code for the demos below is available on GitHub. See the demos along with steps to reproduce:


- [Age and Gender Recognition via REST API](age_gender_guide.md) - run predictions on JPEG images using an age and gender recognition model via the REST API.

- [Horizontal Text Detection in Real-Time](camera_example.md) - run predictions on a camera stream using a horizontal text detection model via the gRPC API.

- [Age, Gender and Emotion Recognition with Pipelined Models](combined_model_dag.md) - run predictions on JPEG images using a pipeline of age-gender recognition and emotion recognition models via the gRPC API.

- [Optical Character Recognition Pipeline](east_ocr.md) - run predictions on JPEG images using a pipeline of text recognition and text detection models, with a custom node for processing intermediate results, via the gRPC API.

- [Simple Face Detection](face_detection_script_example.md) - run predictions on JPEG images using a face detection model via the gRPC API.

- [Age, Gender and Emotion Recognition with Pipelined Models (full)](faces_analysis_dag.md) - run predictions on JPEG images using a pipeline of age-gender recognition and emotion recognition models via the gRPC API.

- [Image Classification with ONNX Models](ovms_onnx_example.md) - run predictions on JPEG images using a ResNet-50 classification model from [ONNX Model Zoo](https://github.com/onnx/models) via the gRPC API.

- [Preparing TensorFlow Model For Serving](tf_model_binary_input.md) - download and convert a [ResNet-50 TensorFlow model](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet) to the [OpenVINO IR](https://docs.openvino.ai/latest/openvino_docs_MO_DG_IR_and_opsets.html#doxid-openvino-docs-m-o-d-g-i-r-and-opsets) format accepted by OpenVINO Model Server.

- [Vehicle Analysis Pipeline](vehicles_analysis_dag.md) - detect vehicles and recognize their attributes using a pipeline of vehicle detection and vehicle attribute recognition models, with a custom node for processing intermediate results, via the gRPC API.

- [Natural Language Processing with BERT](../example_client/bert/README.md) - provide a knowledge source and a query a [BERT model](https://docs.openvino.ai/latest/omz_models_model_bert_small_uncased_whole_word_masking_squad_int8_0002.html) for question answering use case via gRPC API.

- [Run Predictions with a C++ application](../example_client/cpp/README.md) - build a C++ client application in a Docker container and send predictions via the gRPC API. 

- [Run Predictions with a Go application](../example_client/go/README.md) - build a Go client application in a Docker container and send predictions via the gRPC API.

- [ovmsclient Python samples](../client/python/samples/README.md) - a set of samples that use the `ovmsclient` Python package for predictions, getting model status and model metadata via both the gRPC and REST APIs.

Additional demos that show how to handle dynamic inputs:

- [dynamic batch size with a demuliplexer](./dynamic_bs_demultiplexer.md) - create a simple pipeline that splits data of any batch size and performs inference on each element in the batch separately.

- [dynamic batch size with automatic model reloading](./dynamic_bs_auto_reload.md) - configure the model server to reload the model every time it receives a request with batch size other than what is currently set.

- [dynamic shape with automatic model reloading](./dynamic_shape_auto_reload.md) - configure the model server to reload a model every time the model receives a request with data in a shape other than what is currently set.

- [dynamic input shape with a custom node](./dynamic_shape_custom_node.md) - create a simple pipeline by pairing a model with a custom node that performs data preprocessing and provides the model with data in an acceptable shape.

- [dynamic input shape with binary input format](./dynamic_shape_binary_inputs.md) - send data in binary format (i.e. JPEG or PNG encoded), so the model server adjusts the input on data decoding. 