# Clients {#ovms_docs_clients}

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

## Python client

Creating client application in Python is probably the simplest due to existance of two PyPi packages that make developer life easier when developing logic responsible for interaction with OpenVINO Model Server. Namely, the two packages are:
- [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/)
- [ovmsclient](https://pypi.org/project/ovmsclient/)

### Install the package
@sphinxdirective

.. tab:: ovmsclient  

   .. code-block:: sh

        pip3 install ovmsclient 

.. tab:: tensorflow-serving-api  

   .. code-block:: sh  
   
        pip3 install tensorflow-serving-api 

@endsphinxdirective

### Request model status

@sphinxdirective

.. tab:: ovmsclient  

   .. code-block:: python

        from ovmsclient import make_grpc_client
        client = make_grpc_client(service_url="10.20.30.40:9000")
        model_status = client.get_model_status(model_name="my_model")
   
.. tab:: tensorflow-serving-api  

   .. code-block:: python  
   
        import grpc
        from tensorflow_serving.apis import model_service_pb2_grpc, get_model_status_pb2
        channel = grpc.insecure_channel(„10.20.30.40:9000")
        model_service_stub = model_service_pb2_grpc.ModelServiceStub(channel)
        status_request = get_model_status_pb2.GetModelStatusRequest()
        status_request.model_spec.name = "my_model"
        status_response = model_service_stub. GetModelStatus(status_request, 10.0)
     
@endsphinxdirective

### Request model metadata

@sphinxdirective

.. tab:: ovmsclient  

   .. code-block:: python

        from ovmsclient import make_grpc_client
        client = make_grpc_client(service_url="10.20.30.40:9000")
        model_metadata = client.get_model_metadata(model_name="my_model")

.. tab:: tensorflow-serving-api  

   .. code-block:: python  

        import grpc
        from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2
        channel = grpc.insecure_channel(„10.20.30.40:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        metadata_request = get_model_metadata_pb2.GetModelMetadataRequest()
        metadata_request.model_spec.name = "my_model"
        metadata_response = prediction_service_stub.GetModelMetadata(metadata_request, 10.0)

@endsphinxdirective

### Request prediction on binary

@sphinxdirective

.. tab:: ovmsclient  

   .. code-block:: python

        from ovmsclient import make_grpc_client
        client = make_grpc_client(service_url="10.20.30.40:9000")
        with open("img.jpeg", "rb") as f:
            data = f.read()
        inputs = {"input_name": data}    
        results = client.predict(inputs=inputs, model_name="my_model")

.. tab:: tensorflow-serving-api  

   .. code-block:: python  
   
        import grpc
        from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
        from tensorflow import make_tensor_proto
        channel = grpc.insecure_channel(„10.20.30.40:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        with open("img.jpeg", "rb") as f:
            data = f.read()
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = "my_model"
        predict_request.inputs["input_name"].CopyFrom(make_tensor_proto(data))
        predict_response = prediction_service_stub.Predict(predict_request, 10.0)

@endsphinxdirective

### Request prediction on Numpy

@sphinxdirective

.. tab:: ovmsclient  

   .. code-block:: python

        import numpy as np
        from ovmsclient import make_grpc_client
        client = make_grpc_client(service_url="10.20.30.40:9000")
        data = np.array([1.0, 2.0, ..., 1000.0])
        inputs = {"input_name": data}
        results = client.predict(inputs=inputs, model_name="my_model")
   
.. tab:: tensorflow-serving-api  

   .. code-block:: python  
   
        import grpc
        from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
        from tensorflow import make_tensor_proto
        channel = grpc.insecure_channel(„10.20.30.40:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        data = np.array([1.0, 2.0, ..., 1000.0])
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = "my_model"
        predict_request.inputs["input_name"].CopyFrom(make_tensor_proto(data))
        predict_response = prediction_service_stub.Predict(predict_request, 10.0)

@endsphinxdirective

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

## C++ client

Creating client application in C++ follows the same principals as in Python, but in C++ it's a little bit more complicated. There's no package or library with convenient functions for interaction with the model server. 

See https://github.com/openvinotoolkit/model_server/tree/main/example_client/cpp for exemplary code and Dockerfile that takes care of building the application.

## Go client

Creating client application in Go follows the same principals as in Python, but in Go it's a little bit more complicated. There's no package or library with convenient functions for interaction with the model server. 

See https://github.com/openvinotoolkit/model_server/tree/main/example_client/go for exemplary code and Dockerfile that takes care of building the application.