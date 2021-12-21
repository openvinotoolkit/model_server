# Clients {#ovms_docs_clients}

## Python client

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

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
        from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus

        channel = grpc.insecure_channel(„10.20.30.40:9000")
        model_service_stub = model_service_pb2_grpc.ModelServiceStub(channel)

        status_request = get_model_status_pb2.GetModelStatusRequest()
        status_request.model_spec.name = "my_model"
        status_response = model_service_stub. GetModelStatus(status_request, 10.0)
                
        model_status = {}
        model_version_status = status_response.model_version_status
        for model_version in model_version_status:
            model_status[model_version.version] = dict([
                ('state', ModelVersionStatus.State.Name(model_version.state)),
                ('error_code', model_version.status.error_code),
                ('error_message', model_version.status.error_message),
            ])
        
     
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
        from tensorflow.core.framework.types_pb2 import DataType

        channel = grpc.insecure_channel(„10.20.30.40:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        metadata_request = get_model_metadata_pb2.GetModelMetadataRequest()
        metadata_request.model_spec.name = "my_model"
        metadata_response = prediction_service_stub.GetModelMetadata(metadata_request, 10.0)

        model_metadata = {}

        signature_def = metadata_response.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)
        model_signature = signature_map.ListFields()[0][1]['serving_default']

        inputs_metadata = {}
        for input_name, input_info in model_signature.inputs.items():
            input_shape = [d.size for d in input_info.tensor_shape.dim]
            inputs_metadata[input_name] = dict([
                ("shape", input_shape),
                ("dtype", DataType.Name(input_info.dtype))
            ])

        outputs_metadata = {}
        for output_name, output_info in model_signature.outputs.items():
            output_shape = [d.size for d in output_info.tensor_shape.dim]
            outputs_metadata[output_name] = dict([
                ("shape", output_shape),
                ("dtype", DataType.Name(output_info.dtype))
            ])

        version = metadata_response.model_spec.version.value
        model_metadata = dict([
            ("model_version", version),
            ("inputs", inputs_metadata),
            ("outputs", outputs_metadata)
        ])

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
        from tensorflow import make_tensor_proto, make_ndarray

        channel = grpc.insecure_channel(„10.20.30.40:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        with open("img.jpeg", "rb") as f:
            data = f.read()
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = "my_model"
        predict_request.inputs["input_name"].CopyFrom(make_tensor_proto(data))
        predict_response = prediction_service_stub.Predict(predict_request, 10.0)
        results = make_ndarray(predict_response.outputs["output_name"])

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
        results = make_ndarray(predict_response.outputs["output_name"])

@endsphinxdirective

## C++ client

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

Creating client application in C++ follows the same principals as in Python, but in C++ it's a little bit more complicated. There's no package or library with convenient functions for interaction with the model server.

To successfully set up communication with the model server you need to implement the logic to communicate with endpoints specified in the [API](api_reference_guide.md). For gRPC, download and compile protos, and then link and use them in your application according to [gRPC API specification](model_server_grpc_api.md). For REST, prepare your data and pack it to appropriate JSON structure according to [REST API specification](model_server_rest_api.md).

See [C++ demo](../example_client/cpp/README.md) to learn how to build exemplary C++ client application in Docker and use it to run predictions via gRPC API. 

## Go client

@sphinxdirective
.. raw:: html

    <div id="switcher-go" class="switcher-anchor">Go</div>
@endsphinxdirective

Creating client application in Go follows the same principals as in Python, but in Go it's a little bit more complicated. There's no package or library with convenient functions for interaction with the model server.

To successfully set up communication with the model server you need to implement the logic to communicate with endpoints specified in the [API](api_reference_guide.md). For gRPC, download and compile protos, and then link and use them in your application according to [gRPC API specification](model_server_grpc_api.md). For REST, prepare your data and pack it to appropriate JSON structure according to [REST API specification](model_server_rest_api.md).

See [Go demo](../example_client/go/README.md) to learn how to build exemplary Go client application in Docker and use it to run predictions via gRPC API.
