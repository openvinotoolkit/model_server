import numpy as np
from grpc.beta import implementations
import time
import sys
sys.path.append(".")
from ie_serving.tensorflow_serving_api import prediction_service_pb2 # noqa
from conftest import infer # noqa


class TestMuiltModelInference():

    def test_run_inference(self, download_two_models,
                           input_data_downloader_v1_224,
                           input_data_downloader_v3_331,
                           start_server_multi_model):
        """
        <b>Description</b>
        Execute inference request using gRPC interface hosting multiple models

        <b>input data</b>
        - directory with 2 models in IR format
        - docker image
        - input data in numpy format

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape for both models set in config
        file: model resnet_v1_50, pnasnet_large
        - both served models handles appropriate input formats

        """

        print("Downloaded model files:", download_two_models)

        # Starting docker with ie-serving
        result = start_server_multi_model
        print("docker starting multi model server status:", result)
        time.sleep(15)  # Waiting for inference service to load models
        assert result == 0, "docker container was not started successfully"

        # Connect to grpc service
        channel = implementations.insecure_channel('localhost', 9001)
        stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        print("Starting inference using resnet model")
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet_V1_50',
                           model_spec_version=None,
                           output_tensor='resnet_v1_50/predictions/Reshape_1')
            print("output shape", output.shape)
            assert output.shape == (1, 1000), 'resnet model has invalid output'

        imgs_v3_331 = np.array(input_data_downloader_v3_331)
        print("Starting inference using pnasnet_large model")
        for x in range(0, 10):
            output = infer(imgs_v3_331, slice_number=x, input_tensor='input',
                           grpc_stub=stub, model_spec_name='pnasnet_large',
                           model_spec_version=None,
                           output_tensor='final_layer/predictions')
            print("output shape", output.shape)
            assert output.shape == (1, 1001), 'resnet model has invalid output'
