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

- [Vehicle Analysis Pipeline](vehicle_analysis.md) - detect vehicles and recognize their attributes using a pipeline of vehicle detection and vehicle attributes recognition models with a custom node for intermediate results processing via gRPC API.

- [Natural Language Processing with BERT](../example_client/bert/README.md) - provide a knowledge source and a query and use BERT model for question answering use case via gRPC API.