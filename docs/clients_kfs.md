# KServe API Clients {#ovms_docs_clients_kfs}

## Python Client

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

        server_live = client.is_server_live()

        server_ready = client.is_server_ready()

        model_ready = client.is_model_ready("model_name")

.. tab:: python [REST]

    .. code-block:: python

        import tritonclient.http as httpclient

        client = httpclient.InferenceServerClient("localhost:9000")

        server_live = client.is_server_live()

        server_ready = client.is_server_ready()

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

    .. code-block:: cpp

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

    .. code-block:: cpp

        func main() {
            grpc.Dial("localhost:9000", grpc.WithInsecure())
            client := grpc_client.NewGRPCInferenceServiceClient(conn)

            serverLiveRequest := grpc_client.ServerLiveRequest{}
            serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)

            serverReadyRequest := grpc_client.ServerReadyRequest{}
            serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
            
            modelReadyRequest := grpc_client.ModelReadyRequest{
                    Name:    "modelName",
                    Version: "modelVersion",
            }
            modelReadyResponse, err := client.ModelReady(ctx, &modelReadyRequest)
        }

.. tab:: curl    

    .. code-block:: sh  

        curl http://localhost:8000/v2/health/live
        curl http://localhost:8000/v2/health/ready
        curl http://localhost:8000/v2/models/model_name/ready

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

        import tritonclient.http as httpclient

        client = httpclient.InferenceServerClient("localhost:9000")
        server_metadata = client.get_server_metadata()

.. tab:: cpp [GRPC]

    .. code-block:: cpp

        #include "grpc_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerGrpcClient> client;
            tc::InferenceServerGrpcClient::Create(&client, "localhost:9000");

            inference::ServerMetadataResponse server_metadata;
            client->ServerMetadata(&server_metadata);

            std::string name = server_metadata.name();
            std::string version = server_metadata.version();
        }

.. tab:: cpp [REST]

    .. code-block:: cpp

        #include "http_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerHttpClient> client;
            tc::InferenceServerHttpClient::Create(&client, "localhost:9000");

            std::string server_metadata;
            client->ServerMetadata(&server_metadata);
        }

.. tab:: java

    .. code-block:: java

        public static void main(String[] args) {
            ManagedChannel channel = ManagedChannelBuilder
                            .forAddress("localhost", 9000)
                            .usePlaintext().build();
            GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

            ServerMetadataRequest.Builder request = ServerMetadataRequest.newBuilder();
            ServerMetadataResponse response = grpc_stub.serverMetadata(request.build());
            
            channel.shutdownNow();
        }


.. tab:: golang

    .. code-block:: cpp

        grpc.Dial("localhost:9000", grpc.WithInsecure())
        client := grpc_client.NewGRPCInferenceServiceClient(conn)

        serverMetadataRequest := grpc_client.ServerMetadataRequest{}
        serverMetadataResponse, err := client.ServerMetadata(ctx, &serverMetadataRequest)

.. tab:: curl    

    .. code-block:: sh  

        curl http://localhost:8000/v2
        
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

        import tritonclient.http as httpclient

        client = httpclient.InferenceServerClient("localhost:9000")
        model_metadata = client.get_model_metadata("model_name")

.. tab:: cpp [GRPC]

    .. code-block:: cpp

        #include "grpc_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerGrpcClient> client;
            tc::InferenceServerGrpcClient::Create(&client, "localhost:9000");

            inference::ModelMetadataResponse model_metadata;
            client->ModelMetadata(&model_metadata, "model_name", "model_version");
        }

.. tab:: cpp [REST]

    .. code-block:: cpp

        #include "http_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerHttpClient> client;
            tc::InferenceServerHttpClient::Create(&client, "localhost:9000");

            std::string model_metadata;
            client->ModelMetadata(&model_metadata, "model_name", "model_version")
        }

.. tab:: java

    .. code-block:: java

        public static void main(String[] args) {
            ManagedChannel channel = ManagedChannelBuilder
                            .forAddress("localhost", 9000)
                            .usePlaintext().build();
            GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

            ModelMetadataRequest.Builder request = ModelMetadataRequest.newBuilder();
            request.setName("model_name");
            request.setVersion("model_version");
            ModelMetadataResponse response = grpc_stub.modelMetadata(request.build());
            
            channel.shutdownNow();
        }


.. tab:: golang

    .. code-block:: cpp

        grpc.Dial("localhost:9000", grpc.WithInsecure())
        client := grpc_client.NewGRPCInferenceServiceClient(conn)

        modelMetadataRequest := grpc_client.ModelMetadataRequest{
            Name:    "modelName",
            Version: "modelVersion",
        }
        modelMetadataResponse, err := client.ModelMetadata(ctx, &modelMetadataRequest)

.. tab:: curl    

    .. code-block:: sh  

        curl http://localhost:8000/v2/models/model_name
        
@endsphinxdirective

### Request Prediction on an Encoded Image

@sphinxdirective
.. tab:: python [GRPC]

    .. code-block:: python

        from tritonclient.grpc import service_pb2, service_pb2_grpc
        from tritonclient.utils import *

        client = grpcclient.InferenceServerClient("localhost:9000")
        image_data = []
        with open("image_path", 'rb') as f:
            image_data.append(f.read())
        inputs = []
        inputs.append(service_pb2.ModelInferRequest().InferInputTensor())
        inputs[0].name = args['input_name']
        inputs[0].datatype = "BYTES"
        inputs[0].shape.extend([1])
        inputs[0].contents.bytes_contents.append(image_data[0])

        outputs = []
        outputs.append(service_pb2.ModelInferRequest().InferRequestedOutputTensor())
        outputs[0].name = "output_name"

        request = service_pb2.ModelInferRequest()
        request.model_name = "model_name'"
        request.inputs.extend(inputs)

.. tab:: python [REST]

    .. code-block:: python

        import requests
        import json

        url = f"http://{address}/v2/models/{model_name}/infer"
        http_session = requests.session()

        image_data = []
        image_binary_size = []
        with open("image_path", 'rb') as f:
            image_data.append(f.read())
            image_binary_size.append(len(image_data[-1]))
        image_binary_size_str = ",".join(map(str, image_binary_size))
        inference_header = {"inputs":[{"name":input_name,"shape":[batch_i],"datatype":"BYTES","parameters":{"binary_data_size":image_binary_size_str}}]}
        inference_header_binary = json.dumps(inference_header).encode()

        results = http_session.post(url, inference_header_binary + b''.join(image_data), headers={"Inference-Header-Content-Length":str(len(inference_header_binary))})

.. tab:: cpp [GRPC]

    .. code-block:: cpp

        #include "grpc_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerGrpcClient> client;
            tc::InferenceServerGrpcClient::Create(&client, "localhost:9000");

            std::vector<int64_t> shape{1, 10};
            tc::InferInput* input;
            tc::InferInput::Create(&input, "input_name", shape, "FP32");
            std::shared_ptr<tc::InferInput> input_ptr;
            input_ptr.reset(input)

            std::ifstream fileImg("image_path", std::ios::binary);
            fileImg.seekg(0, std::ios::end);
            int bufferLength = fileImg.tellg();
            fileImg.seekg(0, std::ios::beg);

            char* buffer = new char[bufferLength];
            fileImg.read(buffer, bufferLength);

            std::vector<uint8_t> input_data = std::vector<uint8_t>(buffer, buffer + bufferLength);
            input_ptr->AppendRaw(input_data);

            tc::InferOptions options("model_name");
            tc::InferResult* result;
            client->Infer(&result, options, inputs);
            input->Reset();

            delete buffer;
        }

.. tab:: cpp [REST]

    .. code-block:: cpp

        #include "http_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerHttpClient> client;
            tc::InferenceServerHttpClient::Create(&client, "localhost:9000");

            std::vector<int64_t> shape{1};
            tc::InferInput* input;
            tc::InferInput::Create(&input, input_name, shape, "BYTES");
            std::shared_ptr<tc::InferInput> input_ptr;
            input_ptr.reset(input)

            std::ifstream fileImg("image_path", std::ios::binary);
            fileImg.seekg(0, std::ios::end);
            int bufferLength = fileImg.tellg();
            fileImg.seekg(0, std::ios::beg);

            char* buffer = new char[bufferLength];
            fileImg.read(buffer, bufferLength);

            std::vector<uint8_t> input_data = std::vector<uint8_t>(buffer, buffer + bufferLength);
            input_ptr->AppendRaw(input_data);

            tc::InferOptions options("model_name");
            tc::InferResult* result;
            client->Infer(&result, options, inputs);
            input->Reset();

            delete buffer;
        }

.. tab:: java

    .. code-block:: java

        public static void main(String[] args) {
            ManagedChannel channel = ManagedChannelBuilder
                            .forAddress("localhost", 9000)
                            .usePlaintext().build();
            GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

            ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
            request.setModelName("model_name");
            request.setModelVersion("model_version");

            ModelInferRequest.InferInputTensor.Builder input = ModelInferRequest.InferInputTensor
                    .newBuilder();
            String defaultInputName = "b";
            input.setName("input_name");
            input.setDatatype("BYTES");
            input.addShape(1);

            FileInputStream imageStream = new FileInputStream("image_path");
            request.clearRawInputContents();
            request.addRawInputContents(ByteString.readFrom(imageStream));

            ModelInferResponse response = grpc_stub.modelInfer(request.build());
            
            channel.shutdownNow();
        }


.. tab:: golang

    .. code-block:: cpp

        grpc.Dial("localhost:9000", grpc.WithInsecure())
        client := grpc_client.NewGRPCInferenceServiceClient(conn)

        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)

        inferInputs := []*grpc_client.ModelInferRequest_InferInputTensor{
            &grpc_client.ModelInferRequest_InferInputTensor{
                Name:     "0",
                Datatype: "BYTES",
                Shape:    []int64{1},
            },
        }

        bytes, err := ioutil.ReadFile(fileName)
	    modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, bytes) 

        modelInferRequest := grpc_client.ModelInferRequest{
            ModelName:    "model_name",
            ModelVersion: "model_version",
            Inputs:       inferInputs,
        }

        modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)

.. tab:: curl    

    .. code-block:: sh  

        echo -n '{"inputs” : [{"name" : "0", "shape" : [1], "datatype" : "BYTES"}]}' > request.json
        stat --format=%s request.json
        66
        cat ./image.jpeg >> request.json
        curl --data-binary "@./request.json" -X POST http://localhost:8000/v2/models/resnet/versions/0/infer -H "Inference-Header-Content-Length: 66"

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
        import tritonclient.http as httpclient

        client = httpclient.InferenceServerClient("localhost:9000")
        data = np.array([1.0, 2.0, ..., 1000.0])
        infer_input = httpclient.InferInput("input_name", data.shape, "FP32")
        infer_input.set_data_from_numpy(data)
        results = client.infer("model_name", [infer_input]

.. tab:: cpp [GRPC]

    .. code-block:: cpp

        #include "grpc_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerGrpcClient> client;
            tc::InferenceServerGrpcClient::Create(&client, "localhost:9000");

            std::vector<int64_t> shape{1, 10};
            tc::InferInput* input;
            tc::InferInput::Create(&input, "input_name", shape, "FP32");
            std::shared_ptr<tc::InferInput> input_ptr;
            input_ptr.reset(input)

            std::vector<float> input_data(10);
            for (size_t i = 0; i < 10; ++i) {
                input_data[i] = i;
            }
            std::vector<tc::InferInput*> inputs = {input_ptr.get()};
            tc::InferOptions options("model_name");
            tc::InferResult* result;
            input_ptr->AppendRaw(input_data);
            client->Infer(&result, options, inputs);
            input->Reset();
        }

.. tab:: cpp [REST]

    .. code-block:: cpp

        #include "http_client.h"

        namespace tc = triton::client;
        int main() {
            std::unique_ptr<tc::InferenceServerHttpClient> client;
            tc::InferenceServerHttpClient::Create(&client, "localhost:9000");

            std::vector<int64_t> shape{1, 10};
            tc::InferInput* input;
            tc::InferInput::Create(&input, "input_name", shape, "FP32");
            std::shared_ptr<tc::InferInput> input_ptr;
            input_ptr.reset(input)

            std::vector<float> input_data(10);
            for (size_t i = 0; i < 10; ++i) {
                input_data[i] = i;
            }
            std::vector<tc::InferInput*> inputs = {input_ptr.get()};
            tc::InferOptions options("model_name");
            tc::InferResult* result;
            input_ptr->AppendRaw(input_data);
            client->Infer(&result, options, inputs);
            input->Reset();
        }

.. tab:: java

    .. code-block:: java

        public static void main(String[] args) {
            ManagedChannel channel = ManagedChannelBuilder
                            .forAddress("localhost", 9000)
                            .usePlaintext().build();
            GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

            ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
            request.setModelName("model_name");
            request.setModelVersion("model_version");

            List<Float> lst = Arrays.asList(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f);
            InferTensorContents.Builder input_data = InferTensorContents.newBuilder();
            input_data.addAllFp32Contents(lst);

            ModelInferRequest.InferInputTensor.Builder input = ModelInferRequest.InferInputTensor
                    .newBuilder();
            String defaultInputName = "b";
            input.setName("input_name");
            input.setDatatype("FP32");
            input.addShape(1);
            input.addShape(10);
            input.setContents(input_data);

            request.addInputs(0, input);

            ModelInferResponse response = grpc_stub.modelInfer(request.build());
            
            channel.shutdownNow();
        }


.. tab:: golang

    .. code-block:: cpp

        grpc.Dial("localhost:9000", grpc.WithInsecure())
        client := grpc_client.NewGRPCInferenceServiceClient(conn)

        ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)

        inputData := make([]float32, inputSize)
        for i := 0; i < inputSize; i++ {
            inputData[i] = float32(i)
        }

        inferInputs := []*grpc_client.ModelInferRequest_InferInputTensor{
            &grpc_client.ModelInferRequest_InferInputTensor{
                Name:     "b",
                Datatype: "FP32",
                Shape:    []int64{1, 10},
                Contents: &grpc_client.InferTensorContents{
                    Fp32Contents: inputData,
                },
            },
        }

        modelInferRequest := grpc_client.ModelInferRequest{
            ModelName:    "model_name",
            ModelVersion: "model_version",
            Inputs:       inferInputs,
        }

        modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)

.. tab:: curl    

    .. code-block:: sh  

        curl -X POST http://localhost:8000/v2/models/model_name/infer
        -H 'Content-Type: application/json'
        -d '{"inputs" : [ {"name" : "input_name", "shape" : [ 1, 10 ], "datatype"  : "FP32", "data" : [1,2,3,4,5,6,7,8,9,10]} ]}'

@endsphinxdirective

For complete usage examples see [Kserve samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples).
