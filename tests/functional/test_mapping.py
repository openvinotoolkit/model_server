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

import numpy as np
import pytest
from constants import PREDICTION_SERVICE, ERROR_SHAPE
from utils.grpc import infer, get_model_metadata, model_metadata_response
from utils.rest import infer_rest, get_model_metadata_response_rest


class TestSingleModelMappingInference():

    def test_run_inference(self, age_gender_model_downloader,
                           create_grpc_channel,
                           start_server_with_mapping):
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

        print("Downloaded model files:", age_gender_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9002', PREDICTION_SERVICE)

        imgs_v1_224 = np.ones((1, 3, 62, 62))

        output = infer(imgs_v1_224, input_tensor='new_key', grpc_stub=stub,
                       model_spec_name='age_gender',
                       model_spec_version=None,
                       output_tensors=['age', 'gender'])
        print("output shape", output['age'].shape)
        print("output shape", output['gender'].shape)
        assert output['age'].shape == (1, 1, 1, 1), ERROR_SHAPE
        assert output['gender'].shape == (1, 2, 1, 1), ERROR_SHAPE

    def test_get_model_metadata(self, age_gender_model_downloader,
                                create_grpc_channel,
                                start_server_with_mapping):

        print("Downloaded model files:", age_gender_model_downloader)

        stub = create_grpc_channel('localhost:9002', PREDICTION_SERVICE)

        model_name = 'age_gender'
        expected_input_metadata = {'new_key': {'dtype': 1,
                                               'shape': [1, 3, 62, 62]}}
        expected_output_metadata = {'age': {'dtype': 1,
                                            'shape': [1, 1, 1, 1]},
                                    'gender': {'dtype': 1,
                                               'shape': [1, 2, 1, 1]}}
        request = get_model_metadata(model_name=model_name)
        response = stub.GetModelMetadata(request, 10)
        print("response", response)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_rest(self, age_gender_model_downloader,
                                start_server_with_mapping, request_format):
        """
            <b>Description</b>
            Submit request to REST API interface serving a single resnet model

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

        imgs_v1_224 = np.ones((1, 3, 62, 62))
        rest_url = 'http://localhost:5556/v1/models/age_gender:predict'
        output = infer_rest(imgs_v1_224, input_tensor='new_key',
                            rest_url=rest_url,
                            output_tensors=['age', 'gender'],
                            request_format=request_format)
        print("output shape", output['age'].shape)
        print("output shape", output['gender'].shape)
        print(output)
        assert output['age'].shape == (1, 1, 1, 1), ERROR_SHAPE
        assert output['gender'].shape == (1, 2, 1, 1), ERROR_SHAPE

    def test_get_model_metadata_rest(self, age_gender_model_downloader,
                                     start_server_with_mapping):

        print("Downloaded model files:", age_gender_model_downloader)

        model_name = 'age_gender'
        expected_input_metadata = {'new_key': {'dtype': 1,
                                               'shape': [1, 3, 62, 62]}}
        expected_output_metadata = {'age': {'dtype': 1,
                                            'shape': [1, 1, 1, 1]},
                                    'gender': {'dtype': 1,
                                               'shape': [1, 2, 1, 1]}}
        rest_url = 'http://localhost:5556/v1/models/age_gender/metadata'
        response = get_model_metadata_response_rest(rest_url)
        print("response", response)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
