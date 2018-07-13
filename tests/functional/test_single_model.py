import numpy as np
from grpc.beta import implementations
import sys
import time
sys.path.append(".")
from ie_serving.tensorflow_serving_api import prediction_service_pb2 # noqa
from conftest import infer # noqa


class TestSingleModelInference():

    def test_run_inference(self, resnet_v1_50_model_downloader,
                           input_data_downloader_v1_224,
                           start_server_single_model):
        """
        <b>Description</b>
        Submit request to gRPC interface serving a single resnet model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service
        - input data in numpy format

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """

        print("Downloaded model files:", resnet_v1_50_model_downloader)

        # Starting docker with ie-serving
        result = start_server_single_model
        print("docker starting status:", result)
        time.sleep(20)  # Waiting for inference service to load models
        assert result == 0, "docker container was not started successfully"

        # Connect to grpc service
        channel = implementations.insecure_channel('localhost', 9000)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)

        imgs_v1_224 = np.array(input_data_downloader_v1_224)

        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet',
                           model_spec_version=None,
                           output_tensor='resnet_v1_50/predictions/Reshape_1')
        print("output shape", output.shape)
        assert output.shape == (1, 1000), 'resnet model has invalid output'
