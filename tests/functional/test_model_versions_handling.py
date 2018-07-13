import numpy as np
from grpc.beta import implementations
import time
import sys
sys.path.append(".")
from ie_serving.tensorflow_serving_api import prediction_service_pb2 # noqa
from conftest import infer # noqa


class TestModelVersionHandling():

    def test_run_inference(self, download_two_model_versions,
                           input_data_downloader_v1_224,
                           start_server_multi_model):
        """
        <b>Description</b>
        Execute inference request using gRPC interface with version specified
        and without version set on the client.
        When version is not set server should use the latest version model 2
        When version 1 is selected the model from folder 1 should be used
        and model 2 should be ignored

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service
        - input data in numpy format

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - latest model version serves resnet_v2_50 model - [1,1001]
        output resnet_v2_50/predictions/Reshape_1
        - first model version serves resnet_v1_50 model - [1,1000]
        output resnet_v1_50/predictions/Reshape_1
        """

        print("Downloaded model files:", download_two_model_versions)

        # Starting docker with ie-serving
        result = start_server_multi_model
        print("docker starting status:", result)
        time.sleep(20)  # Waiting for inference service to load models
        assert result == 0, "docker container was not started successfully"

        # Connect to grpc service
        channel = implementations.insecure_channel('localhost', 9001)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)

        imgs_v1_224 = np.array(input_data_downloader_v1_224)

        print("Starting inference using latest version - no version set")
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x, input_tensor='input',
                           grpc_stub=stub, model_spec_name='resnet',
                           model_spec_version=None,
                           output_tensor='resnet_v2_50/predictions/Reshape_1')
            print("output shape", output.shape)
            assert output.shape == (1, 1001),\
                'resnet model with version 1 has invalid output'

        # both model versions use the same input data shape
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x, input_tensor='input',
                           grpc_stub=stub, model_spec_name='resnet',
                           model_spec_version=1,
                           output_tensor='resnet_v1_50/predictions/Reshape_1')
            print("output shape", output.shape)
            assert output.shape == (1, 1000),\
                'resnet model with latest version has invalid output'
