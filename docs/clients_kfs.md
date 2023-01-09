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

.. tab:: python [GRPC]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")

                # Check server liveness
                server_live = client.is_server_live()

                # Check server readiness
                server_ready = client.is_server_ready()

                # Check model readiness
                model_ready = client.is_model_ready("model_name")

.. tab:: python [REST]

        .. code-block:: python

                import tritonclient.http as httpclient

                client = httpclient.InferenceServerClient("localhost:9000")

                # Check server liveness
                server_live = client.is_server_live()

                # Check server readiness
                server_ready = client.is_server_ready()

                # Check model readiness
                model_ready = client.is_model_ready("model_name")

.. tab:: cpp [GRPC]

        .. code-block:: cpp

                #include "grpc_client.h"

                namespace tc = triton::client;
                int main() {
                        std::unique_ptr<tc::InferenceServerGrpcClient> client;
                        tc::InferenceServerGrpcClient::Create(&client, "localhost:9000");

                        bool serverLive = client->IsServerLive(&serverLive);

                        bool serverReady = client->IsServerReady(&serverReady);

                        bool modelReady = client->IsModelReady(&modelReady, "model_name", "model_version");
                }

.. tab:: cpp [REST]

        .. code-block:: python

                #include "http_client.h"

                namespace tc = triton::client;
                int main() {
                        std::unique_ptr<tc::InferenceServerHttpClient> client;
                        tc::InferenceServerHttpClient::Create(&client, "localhost:9000");

                        bool serverLive = client->IsServerLive(&serverLive);

                        bool serverReady = client->IsServerReady(&serverReady);

                        bool modelReady = client->IsModelReady(&modelReady, "model_name", "model_version");
                }

.. tab:: java

        .. code-block:: java

                public static void main(String[] args) {
                        ManagedChannel channel = ManagedChannelBuilder
                                        .forAddress("localhost", 9000)
                                        .usePlaintext().build();
                        GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

                        ServerLiveRequest.Builder serverLiveRequest = ServerLiveRequest.newBuilder();
                        ServerLiveResponse serverLiveResponse = grpc_stub.serverLive(serverLiveRequest.build());

                        bool serverLive = serverLiveResponse.getLive();

                        ServerReadyRequest.Builder serverReadyRequest = ServerReadyRequest.newBuilder();
		        ServerReadyResponse serverReadyResponse = grpc_stub.serverReady(serverReadyRequest.build());

                        bool serverReady = serverReadyResponse.getReady();

                        ModelReadyRequest.Builder modelReadyRequest = ModelReadyRequest.newBuilder();
                        modelReadyRequest.setName("model_name");
                        modelReadyRequest.setVersion("version");
                        ModelReadyResponse modelReadyResponse = grpc_stub.modelReady(modelReadyRequest.build());

                        bool modelReady = modelReadyResponse.getReady();
                        
                        channel.shutdownNow();
                }


.. tab:: golang

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")

                # Check server liveness
                server_live = client.is_server_live()

                # Check server readiness
                server_ready = client.is_server_ready()

                # Check model readiness
                model_ready = client.is_model_ready("model_name")

.. tab:: curl    

    .. code-block:: sh  

        curl http://localhost:9000/v2/health/live
        curl http://localhost:9000/v2/health/ready
        curl http://localhost:9000/v2

@endsphinxdirective

### Request Server Metadata

@sphinxdirective

.. tab:: python [GRPC]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                server_metadata = client.get_server_metadata()

.. tab:: python [REST]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                server_metadata = client.get_server_metadata()

.. tab:: cpp [GRPC]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                server_metadata = client.get_server_metadata()

.. tab:: cpp [REST]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                server_metadata = client.get_server_metadata()

.. tab:: java

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                server_metadata = client.get_server_metadata()

.. tab:: golang

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                server_metadata = client.get_server_metadata()

.. tab:: curl

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                server_metadata = client.get_server_metadata()
        
@endsphinxdirective

### Request Model Metadata

@sphinxdirective
.. tab:: python [GRPC]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                model_metadata = client.get_model_metadata("model_name")
.. tab:: python [REST]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                model_metadata = client.get_model_metadata("model_name")

.. tab:: cpp [GRPC]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                model_metadata = client.get_model_metadata("model_name")

.. tab:: cpp [REST]

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                model_metadata = client.get_model_metadata("model_name")

.. tab:: java

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                model_metadata = client.get_model_metadata("model_name")

.. tab:: golang

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                model_metadata = client.get_model_metadata("model_name")

.. tab:: curl

        .. code-block:: python

                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                model_metadata = client.get_model_metadata("model_name")
        
@endsphinxdirective

### Request Prediction on a Numpy Array

@sphinxdirective
.. tab:: python [GRPC]

        .. code-block:: python

                import numpy as np
                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                data = np.array([1.0, 2.0, ..., 1000.0])
                infer_input = grpcclient.InferInput("input_name", data.shape, "FP32")
                infer_input.set_data_from_numpy(data)
                results = client.infer("model_name", [infer_input])

.. tab:: python [REST]

        .. code-block:: python

                import numpy as np
                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                data = np.array([1.0, 2.0, ..., 1000.0])
                infer_input = grpcclient.InferInput("input_name", data.shape, "FP32")
                infer_input.set_data_from_numpy(data)
                results = client.infer("model_name", [infer_input])

.. tab:: cpp [GRPC]

        .. code-block:: python

                import numpy as np
                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                data = np.array([1.0, 2.0, ..., 1000.0])
                infer_input = grpcclient.InferInput("input_name", data.shape, "FP32")
                infer_input.set_data_from_numpy(data)
                results = client.infer("model_name", [infer_input])

.. tab:: cpp [REST]

        .. code-block:: python

                import numpy as np
                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                data = np.array([1.0, 2.0, ..., 1000.0])
                infer_input = grpcclient.InferInput("input_name", data.shape, "FP32")
                infer_input.set_data_from_numpy(data)
                results = client.infer("model_name", [infer_input])

.. tab:: java

        .. code-block:: python

                import numpy as np
                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                data = np.array([1.0, 2.0, ..., 1000.0])
                infer_input = grpcclient.InferInput("input_name", data.shape, "FP32")
                infer_input.set_data_from_numpy(data)
                results = client.infer("model_name", [infer_input])

.. tab:: golang

        .. code-block:: python

                import numpy as np
                import tritonclient.grpc as grpcclient

                client = grpcclient.InferenceServerClient("localhost:9000")
                data = np.array([1.0, 2.0, ..., 1000.0])
                infer_input = grpcclient.InferInput("input_name", data.shape, "FP32")
                infer_input.set_data_from_numpy(data)
                results = client.infer("model_name", [infer_input])

@endsphinxdirective

For complete usage examples see [Kserve samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples).
