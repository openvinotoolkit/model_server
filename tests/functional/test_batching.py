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
import numpy as np
from constants import PREDICTION_SERVICE, ERROR_SHAPE
from utils.grpc import infer, get_model_metadata, \
    model_metadata_response
from utils.rest import infer_rest, get_model_metadata_response_rest


class TestBatchModelInference():

    def test_run_inference(self, resnet_multiple_batch_sizes,
                           start_server_batch_model,
                           create_grpc_channel):
        """
        <b>Description</b>
        Submit request to gRPC interface serving a single resnet model

        <b>input data</b>
        - directory with the model in IR format
        - docker image with ie-serving-py service

        <b>fixtures used</b>
        - model downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape

        """

        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9003', PREDICTION_SERVICE)

        batch_input = np.ones((8, 3, 224, 224))
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        output = infer(batch_input, input_tensor=in_name,
                       grpc_stub=stub, model_spec_name='resnet',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (8, 1001), ERROR_SHAPE

    def test_run_inference_bs4(self, resnet_multiple_batch_sizes,
                               start_server_batch_model_bs4,
                               create_grpc_channel):

        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9004', PREDICTION_SERVICE)

        batch_input = np.ones((4, 3, 224, 224))
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        output = infer(batch_input, input_tensor=in_name,
                       grpc_stub=stub, model_spec_name='resnet',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (4, 1001), ERROR_SHAPE

    def test_run_inference_auto(self, resnet_multiple_batch_sizes,
                                start_server_batch_model_auto,
                                create_grpc_channel):

        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9005', PREDICTION_SERVICE)

        batch_input = np.ones((6, 3, 224, 224))
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        output = infer(batch_input, input_tensor=in_name,
                       grpc_stub=stub, model_spec_name='resnet',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (6, 1001), ERROR_SHAPE

        batch_input = np.ones((1, 3, 224, 224))
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        output = infer(batch_input, input_tensor=in_name,
                       grpc_stub=stub, model_spec_name='resnet',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1001), ERROR_SHAPE

    def test_get_model_metadata(self, resnet_multiple_batch_sizes,
                                start_server_batch_model,
                                create_grpc_channel):

        print("Downloaded model files:", resnet_multiple_batch_sizes)

        stub = create_grpc_channel('localhost:9003', PREDICTION_SERVICE)

        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata = {in_name:   {'dtype': 1,
                                               'shape': [8, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [8, 1001]}}
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
    def test_run_inference_rest(self, age_gender_model_downloader,
                                start_server_batch_model_2out, request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving
            a single age-gender model with 2 outputs.
            No batch_size parameter specified.

            <b>input data</b>
            - directory with the model in IR format
            - docker image with ie-serving-py service

            <b>fixtures used</b>
            - model downloader
            - service launching

            <b>Expected results</b>
            - response contains proper numpy shape

        """

        print("Downloaded model files:", age_gender_model_downloader)

        batch_input = np.ones((1, 3, 62, 62))
        in_name = 'data'
        out_names = ['age_conv3', 'prob']
        rest_url = 'http://localhost:5560/v1/models/age_gender:predict'
        output = infer_rest(batch_input, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        assert output[out_names[0]].shape == (1, 1, 1, 1), ERROR_SHAPE
        assert output[out_names[1]].shape == (1, 2, 1, 1), ERROR_SHAPE

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_bs4_rest(self, age_gender_model_downloader,
                                    start_server_batch_model_auto_bs4_2out,
                                    request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving
            a single age-gender model with 2 outputs.
            Parameter batch_size explicitly set to 4.

            <b>input data</b>
            - directory with the model in IR format
            - docker image with ie-serving-py service

            <b>fixtures used</b>
            - model downloader
            - service launching

            <b>Expected results</b>
            - response contains proper numpy shape

        """

        print("Downloaded model files:", age_gender_model_downloader)

        batch_input = np.ones((4, 3, 62, 62))
        in_name = 'data'
        out_names = ['age_conv3', 'prob']
        rest_url = 'http://localhost:5562/v1/models/age_gender:predict'
        output = infer_rest(batch_input, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        assert output[out_names[0]].shape == (4, 1, 1, 1), ERROR_SHAPE
        assert output[out_names[1]].shape == (4, 2, 1, 1), ERROR_SHAPE

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_rest_auto(self, age_gender_model_downloader,
                                     start_server_batch_model_auto_2out,
                                     request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving a single resnet model.
            Parameter batch_size set to auto.

            <b>input data</b>
            - directory with the model in IR format
            - docker image with ie-serving-py service

            <b>fixtures used</b>
            - model downloader
            - service launching

            <b>Expected results</b>
            - response contains proper numpy shape

        """

        print("Downloaded model files:", age_gender_model_downloader)
        batch_input = np.ones((6, 3, 62, 62))
        in_name = 'data'
        out_names = ['age_conv3', 'prob']
        rest_url = 'http://localhost:5561/v1/models/age_gender:predict'
        output = infer_rest(batch_input,
                            input_tensor=in_name, rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        assert output[out_names[0]].shape == (6, 1, 1, 1), ERROR_SHAPE
        assert output[out_names[1]].shape == (6, 2, 1, 1), ERROR_SHAPE

        batch_input = np.ones((3, 3, 62, 62))
        output = infer_rest(batch_input, input_tensor=in_name,
                            rest_url=rest_url,
                            output_tensors=out_names,
                            request_format=request_format)
        assert output[out_names[0]].shape == (3, 1, 1, 1), ERROR_SHAPE
        assert output[out_names[1]].shape == (3, 2, 1, 1), ERROR_SHAPE

    def test_get_model_metadata_rest(self, resnet_multiple_batch_sizes,
                                     start_server_batch_model):

        print("Downloaded model files:", resnet_multiple_batch_sizes)

        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata = {in_name:   {'dtype': 1,
                                               'shape': [8, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [8, 1001]}}
        rest_url = 'http://localhost:5557/v1/models/resnet/metadata'
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
