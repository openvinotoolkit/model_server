# KServe API Clients {#ovms_docs_clients_kfs}

## Python Client

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

When creating a Python-based client application, you can use Triton client library - [tritonclient](https://pypi.org/project/tritonclient/).

### Install the Package
@sphinxdirective

.. code-block:: sh

        pip3 install tritonclient[all] 

@endsphinxdirective

### Request Health Endpoints

@sphinxdirective

.. code-block:: python

        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient("localhost:9000")

        # Check server liveness
        server_live = client.is_server_live()

        # Check server readiness
        server_ready = client.is_server_ready()

        # Check model readiness
        model_ready = client.is_model_ready("model_name")


@endsphinxdirective

### Request Server Metadata

@sphinxdirective

.. code-block:: python

        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient("localhost:9000")
        server_metadata = client.get_server_metadata()
        
@endsphinxdirective

### Request Model Metadata

@sphinxdirective

.. code-block:: python

        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient("localhost:9000")
        model_metadata = client.get_model_metadata("model_name")
        
@endsphinxdirective

### Request Prediction on a Numpy Array

@sphinxdirective

.. code-block:: python

        import numpy as np
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient("localhost:9000")
        data = np.array([1.0, 2.0, ..., 1000.0])
        infer_input = grpcclient.InferInput("input_name", data.shape, "FP32")
        infer_input.set_data_from_numpy(data)
        results = client.infer("model_name", [infer_input])

@endsphinxdirective

For complete usage examples see [Kserve samples](https://github.com/openvinotoolkit/model_server/tree/v2022.3/client/python/kserve-api/samples).

## C++ Client

@sphinxdirective
.. raw:: html
    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

Creating a client application in C++ follows the same principles as Python. When creating a C++-based client application, you can use Triton client library - [tritonclient](https://github.com/triton-inference-server/client).

See our [C++ samples](https://github.com/openvinotoolkit/model_server/tree/v2022.3/client/cpp/kserve-api/README.md) to learn how to build a sample C++ client application. 
