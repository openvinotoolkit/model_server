import numpy as np
import time
import sys
sys.path.append(".")
from conftest import infer, get_model_metadata, model_metadata_response # noqa


class TestModelVersionHandling():

    def test_run_inference(self, download_two_model_versions,
                           input_data_downloader_v1_224,
                           start_server_multi_model,
                           create_channel_for_port_multi_server):
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
        time.sleep(60)  # Waiting for inference service to load models
        assert result == 0, "docker container was not started successfully"

        # Connect to grpc service
        stub = create_channel_for_port_multi_server

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        out_name_v1 = 'resnet_v1_50/predictions/Reshape_1'
        out_name_v2 = 'resnet_v2_50/predictions/Reshape_1'
        print("Starting inference using latest version - no version set")
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x, input_tensor='input',
                           grpc_stub=stub, model_spec_name='resnet',
                           model_spec_version=None,
                           output_tensors=[out_name_v2])
            print("output shape", output[out_name_v2].shape)
            assert output[out_name_v2].shape == (1, 1001),\
                'resnet model with version 1 has invalid output'

        # both model versions use the same input data shape
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x, input_tensor='input',
                           grpc_stub=stub, model_spec_name='resnet',
                           model_spec_version=1,
                           output_tensors=[out_name_v1])
            print("output shape", output[out_name_v1].shape)
            assert output[out_name_v1].shape == (1, 1000),\
                'resnet model with latest version has invalid output'

    def test_get_model_metadata(self, download_two_models,
                                start_server_multi_model,
                                create_channel_for_port_multi_server):
        """
        <b>Description</b>
        Execute inference request using gRPC interface hosting multiple models

        <b>input data</b>
        - directory with 2 models in IR format
        - docker image

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper response about model metadata for both
        models set in config file:
        model resnet_v1_50, pnasnet_large
        - both served models handles appropriate input formats

        """
        print("Downloaded model files:", download_two_models)

        # Starting docker with ie-serving
        result = start_server_multi_model
        print("docker starting multi model server status:", result)
        time.sleep(30)  # Waiting for inference service to load models
        assert result == 0, "docker container was not started successfully"

        # Connect to grpc service
        stub = create_channel_for_port_multi_server
        versions = [None, 1]
        output_names = ['resnet_v2_50/predictions/Reshape_1',
                        'resnet_v1_50/predictions/Reshape_1']
        for x in range(len(versions)):
            print("Getting info about resnet model version:".format(
                versions[x]))
            model_name = 'resnet'
            out_name = output_names[x]
            expected_input_metadata = {'input': {'dtype': 1,
                                                 'shape': [1, 3, 224, 224]}}
            expected_output_metadata = {out_name: {'dtype': 1,
                                                   'shape': [1, 1, 1]}}
            request = get_model_metadata(model_name='resnet',
                                         version=versions[x])
            response = stub.GetModelMetadata(request, 10)
            input_metadata, output_metadata = model_metadata_response(
                response=response)

            print(output_metadata)
            assert model_name == response.model_spec.name
            assert expected_input_metadata == input_metadata
            assert expected_output_metadata == output_metadata
