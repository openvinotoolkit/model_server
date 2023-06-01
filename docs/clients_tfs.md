# TensorFlow Serving API Clients {#ovms_docs_clients_tfs}

@sphinxdirective

-  `Python Client <#-python-client>`__
-  `C++ and Go Clients <#-cpp-go-clients>`__

.. raw:: html

   <a name='-python-client' id='-python-client'/>

`Python Client`_
================

@endsphinxdirective

When creating a Python-based client application, there are two packages on PyPi that can be used with OpenVINO Model Server:
- [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/)
- [ovmsclient](https://pypi.org/project/ovmsclient/)

### Install the Package
@sphinxdirective

.. tab:: ovmsclient  

    .. code-block:: sh

        pip3 install ovmsclient 

.. tab:: tensorflow-serving-api  

    .. code-block:: sh  

        pip3 install tensorflow-serving-api 

@endsphinxdirective

### Request Model Status

@sphinxdirective

.. tab:: ovmsclient [GRPC]

    .. code-block:: python

        from ovmsclient import make_grpc_client

        client = make_grpc_client("localhost:9000")
        status = client.get_model_status(model_name="my_model")


.. tab:: ovmsclient [REST]

    .. code-block:: python

        from ovmsclient import make_http_client

        client = make_http_client("localhost:8000")
        status = client.get_model_status(model_name="my_model")


.. tab:: tensorflow-serving-api

    .. code-block:: python

        import grpc
        from tensorflow_serving.apis import model_service_pb2_grpc, get_model_status_pb2
        from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus

        channel = grpc.insecure_channel("localhost:9000")
        model_service_stub = model_service_pb2_grpc.ModelServiceStub(channel)

        status_request = get_model_status_pb2.GetModelStatusRequest()
        status_request.model_spec.name = "my_model"
        status_response = model_service_stub.GetModelStatus(status_request, 10.0)
                
        model_status = {}
        model_version_status = status_response.model_version_status
        for model_version in model_version_status:
            model_status[model_version.version] = dict([
                ('state', ModelVersionStatus.State.Name(model_version.state)),
                ('error_code', model_version.status.error_code),
                ('error_message', model_version.status.error_message),
            ])

.. tab:: curl    

    .. code-block:: sh  

        curl http://localhost:8000/v1/models/my_model
    
@endsphinxdirective

### Request Model Metadata

@sphinxdirective

.. tab:: ovmsclient [GRPC]

    .. code-block:: python

        from ovmsclient import make_grpc_client

        client = make_grpc_client("localhost:9000")
        model_metadata = client.get_model_metadata(model_name="my_model")


.. tab:: ovmsclient [REST]

    .. code-block:: python

        from ovmsclient import make_http_client

        client = make_http_client("localhost:8000")
        model_metadata = client.get_model_metadata(model_name="my_model")


.. tab:: tensorflow-serving-api

    .. code-block:: python

        import grpc
        from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2
        from tensorflow.core.framework.types_pb2 import DataType

        channel = grpc.insecure_channel("localhost:9000")
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

.. tab:: curl

    .. code-block:: sh  

        curl http://localhost:8000/v1/models/my_model/metadata


@endsphinxdirective

### Request Prediction on a Binary Input

@sphinxdirective

.. tab:: ovmsclient [GRPC]

    .. code-block:: python

        from ovmsclient import make_grpc_client

        client = make_grpc_client("localhost:9000")
        with open("img.jpeg", "rb") as f:
            data = f.read()
        inputs = {"input_name": data}    
        results = client.predict(inputs=inputs, model_name="my_model")

.. tab:: ovmsclient [REST]

    .. code-block:: python

        from ovmsclient import make_http_client

        client = make_http_client("localhost:8000")

        with open("img.jpeg", "rb") as f:
            data = f.read()
        inputs = {"input_name": data}    
        results = client.predict(inputs=inputs, model_name="my_model")


.. tab:: tensorflow-serving-api

    .. code-block:: python

        import grpc
        from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
        from tensorflow import make_tensor_proto, make_ndarray

        channel = grpc.insecure_channel("localhost:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        with open("img.jpeg", "rb") as f:
            data = f.read()
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = "my_model"
        predict_request.inputs["input_name"].CopyFrom(make_tensor_proto(data))
        predict_response = prediction_service_stub.Predict(predict_request, 10.0)
        results = make_ndarray(predict_response.outputs["output_name"])

.. tab:: curl

    .. code-block:: sh  

        curl -X POST http://localhost:8000/v1/models/my_model:predict
        -H 'Content-Type: application/json'
        -d '{"instances": [{"input_name": {"b64":"YXdlc29tZSBpbWFnZSBieXRlcw=="}}]}'

@endsphinxdirective

### Request Prediction on a Numpy Array

@sphinxdirective

.. tab:: ovmsclient [GRPC]

    .. code-block:: python

        import numpy as np
        from ovmsclient import make_grpc_client

        client = make_grpc_client("localhost:9000")
        data = np.array([1.0, 2.0, ..., 1000.0])
        inputs = {"input_name": data}
        results = client.predict(inputs=inputs, model_name="my_model")

.. tab:: ovmsclient [REST]

    .. code-block:: python

        import numpy as np
        from ovmsclient import make_http_client

        client = make_http_client("localhost:8000")

        data = np.array([1.0, 2.0, ..., 1000.0])
        inputs = {"input_name": data}
        results = client.predict(inputs=inputs, model_name="my_model")

.. tab:: tensorflow-serving-api  

    .. code-block:: python

        import grpc
        from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
        from tensorflow import make_tensor_proto

        channel = grpc.insecure_channel("localhost:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        data = np.array([1.0, 2.0, ..., 1000.0])
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = "my_model"
        predict_request.inputs["input_name"].CopyFrom(make_tensor_proto(data))
        predict_response = prediction_service_stub.Predict(predict_request, 10.0)
        results = make_ndarray(predict_response.outputs["output_name"])

.. tab:: curl

    .. code-block:: sh  

        curl -X POST http://localhost:8000/v1/models/my_model:predict
        -H 'Content-Type: application/json'
        -d '{"instances": [{"input_name": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]}'

@endsphinxdirective

### Request Prediction on a string

@sphinxdirective

.. tab:: ovmsclient [GRPC]

    .. code-block:: python

        from ovmsclient import make_grpc_client

        client = make_grpc_client("localhost:9000")
        data = ["<string>"]
        inputs = {"input_name": data}
        results = client.predict(inputs=inputs, model_name="my_model")

.. tab:: ovmsclient [REST]

    .. code-block:: python

        from ovmsclient import make_http_client

        client = make_http_client("localhost:8000")

        data = ["<string>"]
        inputs = {"input_name": data}
        results = client.predict(inputs=inputs, model_name="my_model")

.. tab:: tensorflow-serving-api  

    .. code-block:: python

        import grpc
        from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
        from tensorflow import make_tensor_proto

        channel = grpc.insecure_channel("localhost:9000")
        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        data = ["<string>"]
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = "my_model"
        predict_request.inputs["input_name"].CopyFrom(make_tensor_proto(data))
        predict_response = prediction_service_stub.Predict(predict_request, 1)
        results = predict_response.outputs["output_name"]

.. tab:: curl

    .. code-block:: sh  

        curl -X POST http://localhost:8000/v1/models/my_model:predict
        -H 'Content-Type: application/json'
        -d '{"instances": [{"input_name": "<string>"}]}'

@endsphinxdirective
For complete usage examples see [ovmsclient samples](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/client/python/ovmsclient/samples).

@sphinxdirective

.. raw:: html

   <a name='-cpp-go-clients' id='-cpp-go-clients'/>

`C++ and Go Clients`_
=====================

@endsphinxdirective

Creating a client application in C++ or [Go](https://go.dev/) follows the same principles as Python, but using them adds some complexity. There is no package or library available for them with convenient functions to interact with OpenVINO Model Server.

To successfully set up communication with the model server, you need to implement the logic to communicate with endpoints specified in the [API](api_reference_guide.md). For gRPC, download and compile protos, then link and use them in your application according to the [gRPC API specification](model_server_grpc_api_tfs.md). For REST, prepare your data and pack it into the appropriate JSON structure according to the [REST API specification](model_server_rest_api_tfs.md).

See our [C++ demo](../demos/image_classification/cpp/README.md) or [Go demo](../demos/image_classification/go/README.md) to learn how to build a sample C++ and Go-based client application in a Docker container and get predictions via the gRPC API. 
