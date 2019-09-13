#
# Copyright (c) 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
from constants import PREDICTION_SERVICE, ERROR_SHAPE
from utils.grpc import infer_batch, get_model_metadata, \
    model_metadata_response
from utils.rest import infer_batch_rest, get_model_metadata_response_rest


class TestBatchModelInference():

    def test_run_inference(self, resnet_8_batch_model_downloader,
                           input_data_downloader_v1_224,
                           start_server_batch_model,
                           create_grpc_channel):
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

        print("Downloaded model files:", resnet_8_batch_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9003', PREDICTION_SERVICE)

        batch_input = input_data_downloader_v1_224[:8, :, :, :]
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        output = infer_batch(batch_input=batch_input, input_tensor='input',
                             grpc_stub=stub, model_spec_name='resnet',
                             model_spec_version=None,
                             output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (8, 1000), ERROR_SHAPE

    def test_run_inference_bs4(self, resnet_8_batch_model_downloader,
                               input_data_downloader_v1_224,
                               start_server_batch_model_bs4,
                               create_grpc_channel):

        print("Downloaded model files:", resnet_8_batch_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9004', PREDICTION_SERVICE)

        batch_input = input_data_downloader_v1_224[:4, :, :, :]
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        output = infer_batch(batch_input=batch_input, input_tensor='input',
                             grpc_stub=stub, model_spec_name='resnet',
                             model_spec_version=None,
                             output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (4, 1000), ERROR_SHAPE

    def test_run_inference_auto(self, resnet_8_batch_model_downloader,
                                input_data_downloader_v1_224,
                                start_server_batch_model_auto,
                                create_grpc_channel):

        print("Downloaded model files:", resnet_8_batch_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9005', PREDICTION_SERVICE)

        batch_input = input_data_downloader_v1_224[:6, :, :, :]
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        output = infer_batch(batch_input=batch_input, input_tensor='input',
                             grpc_stub=stub, model_spec_name='resnet',
                             model_spec_version=None,
                             output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (6, 1000), ERROR_SHAPE

        batch_input = input_data_downloader_v1_224[:1, :, :, :]
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        output = infer_batch(batch_input=batch_input, input_tensor='input',
                             grpc_stub=stub, model_spec_name='resnet',
                             model_spec_version=None,
                             output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1000), ERROR_SHAPE

    def test_get_model_metadata(self, resnet_8_batch_model_downloader,
                                start_server_batch_model,
                                create_grpc_channel):

        print("Downloaded model files:", resnet_8_batch_model_downloader)

        stub = create_grpc_channel('localhost:9003', PREDICTION_SERVICE)

        model_name = 'resnet'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [8, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [8, 1000]}}
        request = get_model_metadata(model_name='resnet')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_rest(self, resnet_2_out_model_downloader,
                                input_data_downloader_v1_224,
                                start_server_batch_model, request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving a single resnet model

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

        print("Downloaded model files:", resnet_2_out_model_downloader)

        batch_input = input_data_downloader_v1_224[:8, :, :, :]
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        rest_url = 'http://localhost:5557/v1/models/resnet:predict'
        output = infer_batch_rest(batch_input,
                                  input_tensor='input', rest_url=rest_url,
                                  output_tensors=[out_name],
                                  request_format=request_format)
        assert output[out_name].shape == (8, 1000), ERROR_SHAPE

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_bs4_rest(self, resnet_2_out_model_downloader,
                                    input_data_downloader_v1_224,
                                    start_server_batch_model_bs4,
                                    request_format):
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

        print("Downloaded model files:", resnet_2_out_model_downloader)

        batch_input = input_data_downloader_v1_224[:4, :, :, :]
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        rest_url = 'http://localhost:5558/v1/models/resnet:predict'
        output = infer_batch_rest(batch_input,
                                  input_tensor='input', rest_url=rest_url,
                                  output_tensors=[out_name],
                                  request_format=request_format)
        assert output[out_name].shape == (4, 1000), ERROR_SHAPE

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_rest_auto(self, resnet_2_out_model_downloader,
                                     input_data_downloader_v1_224,
                                     start_server_batch_model_auto,
                                     request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving a single resnet model

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

        print("Downloaded model files:", resnet_2_out_model_downloader)
        batch_input = input_data_downloader_v1_224[:6, :, :, :]
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        rest_url = 'http://localhost:5559/v1/models/resnet:predict'
        output = infer_batch_rest(batch_input,
                                  input_tensor='input', rest_url=rest_url,
                                  output_tensors=[out_name],
                                  request_format=request_format)
        assert output[out_name].shape == (6, 1000), ERROR_SHAPE

        batch_input = input_data_downloader_v1_224[:1, :, :, :]
        output = infer_batch_rest(batch_input,
                                  input_tensor='input', rest_url=rest_url,
                                  output_tensors=[out_name],
                                  request_format=request_format)
        assert output[out_name].shape == (1, 1000), ERROR_SHAPE

    def test_get_model_metadata_rest(self, resnet_8_batch_model_downloader,
                                     start_server_batch_model):

        print("Downloaded model files:", resnet_8_batch_model_downloader)

        model_name = 'resnet'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [8, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [8, 1000]}}
        rest_url = 'http://localhost:5557/v1/models/resnet/metadata'
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
