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

OpenVINO Model Server demos have been created to showcase the usage of the model server as well as demonstrate it's capabilities. Source codes for all of below demos are available on GitHub. See the demos along with steps to reproduce:


- [Age and Gender Recognition via REST API](age_gender_guide.md) - run prediction on a JPEG image using an age and gender recognition model via REST API.

- [Horizontal Text Detection in Real-Time](camera_example.md) - run prediction on camera stream using a horizontal text detection model via gRPC API.

- [Age, Gender and Emotion Recognition with Pipelined Models](combined_model_dag.md) - run prediction on a JPEG image using a pipeline of age-gender recognition and emotion recognition models via gRPC API.

- [Optical Character Recognition Pipeline](east_ocr.md) - run prediction on a JPEG image using a pipeline of text recognition and text detection models with a custom node for intermediate results processing via gRPC API.

- [Simple Face Detection](face_detection_script_example.md) - run prediction on a JPEG image using a face detection model via gRPC API.

- [Age, Gender and Emotion Recognition with Pipelined Models (full)](faces_analysis_dag.md) - run prediction on a JPEG image using a pipeline of age-gender recognition and emotion recogition models via gRPC API.

- [Image Classification with ONNX Model](ovms_onnx_example.md) - run prediction on a JPEG image using an original ONNX classificatin model via gRPC API.

- [Preparing TensorFlow Model For Serving](tf_model_binary_input.md) - download and convert original TensorFlow model to the format accepted by OpenVINO Model Server.

- [Vehicle Analysis Pipeline](vehicles_analysis_dag.md) - detect vehicles and recognize their attributes using a pipeline of vehicle detection and vehicle attributes recognition models with a custom node for intermediate results processing via gRPC API.

- [Natural Language Processing with BERT](../example_client/bert/README.md) - provide a knowledge source and a query and use BERT model for question answering use case via gRPC API.

- [Run Predictions with C++ application](../example_client/cpp/README.md) - build C++ client application in Docker and use it to run predictions via gRPC API. 

- [Run Predictions with Go application](../example_client/go/README.md) - build Go client application in Docker and use it to run predictions via gRPC API.

- [ovmsclient package examples](../client/python/samples/README.md) - use a set of samples based on `ovmsclient` Python package for predictions, getting model status and getting model metadata both via gRPC and REST APIs

Additional demos that show how to work with dynamic inputs:

- [Handling dynamic batch size with a demuliplexer](./dynamic_bs_demultiplexer.md) - create a simple pipeline that will split data of any batch size and perform inference on each element in the batch separately.

- [Handling dynamic batch size with automatic model reloading](./dynamic_bs_auto_reload.md) - configure model server to reload the model every time it receives a request with batch size other than currently set.

- [Handling dynamic shape with automatic model reloading](./dynamic_shape_auto_reload.md) - configure model server to reload the model every time model receives a request with data in shape other than currently set.

- [Handling dynamic input shape with a custom node](./dynamic_shape_custom_node.md) - create a simple pipeline by pairing your model with a custom node that will perform data preprocessing and provide your model with data in acceptable shape.

- [Handling dynamic input shape with binary input format](./dynamic_shape_binary_inputs.md) - send data in binary format (JPEG or PNG encoded), so model server will adjust the input on data decoding. 