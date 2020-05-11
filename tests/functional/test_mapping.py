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
from constants import ERROR_SHAPE
from model.models_information import AgeGender
from utils.grpc import create_channel, infer, get_model_metadata, model_metadata_response
from utils.rest import infer_rest, get_model_metadata_response_rest


class TestSingleModelMappingInference:

    def test_run_inference(self, age_gender_model_downloader,
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

        _, ports = start_server_with_mapping
        print("Downloaded model files:", age_gender_model_downloader)

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        imgs_v1_224 = np.ones(AgeGender.input_shape, AgeGender.dtype)

        output = infer(imgs_v1_224, input_tensor=AgeGender.input_name, grpc_stub=stub,
                       model_spec_name=AgeGender.name,
                       model_spec_version=None,
                       output_tensors=AgeGender.output_name)
        for output_name, shape in AgeGender.output_shape.items():
            print("output shape", output[output_name].shape)
            assert output[output_name].shape == shape, ERROR_SHAPE

    def test_get_model_metadata(self, age_gender_model_downloader,
                                start_server_with_mapping):

        _, ports = start_server_with_mapping
        print("Downloaded model files:", age_gender_model_downloader)

        stub = create_channel(port=ports["grpc_port"])

        expected_input_metadata = {AgeGender.input_name: {'dtype': 1, 'shape': list(AgeGender.input_shape)}}
        expected_output_metadata = {}
        for output_name, shape in AgeGender.output_shape.items():
            expected_output_metadata[output_name] = {'dtype': 1, 'shape': list(shape)}
        request = get_model_metadata(model_name=AgeGender.name)
        response = stub.GetModelMetadata(request, 10)
        print("response", response)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        assert response.model_spec.name == AgeGender.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.skip(reason="not implemented yet")
    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
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

        _, ports = start_server_with_mapping
        print("Downloaded model files:", age_gender_model_downloader)

        imgs_v1_224 = np.ones(AgeGender.input_shape, AgeGender.dtype)
        rest_url = 'http://localhost:{}/v1/models/age_gender:predict'.format(
                   ports["rest_port"])
        output = infer_rest(imgs_v1_224, input_tensor=AgeGender.input_name,
                            rest_url=rest_url,
                            output_tensors=AgeGender.output_name,
                            request_format=request_format)
        print(output)
        for output_name, shape in AgeGender.output_shape.items():
            print("output shape", output[output_name].shape)
            assert output[output_name].shape == shape, ERROR_SHAPE

    @pytest.mark.skip(reason="not implemented yet")
    def test_get_model_metadata_rest(self, age_gender_model_downloader,
                                     start_server_with_mapping):

        _, ports = start_server_with_mapping
        print("Downloaded model files:", age_gender_model_downloader)

        expected_input_metadata = {AgeGender.input_name: {'dtype': 1, 'shape': list(AgeGender.input_shape)}}
        expected_output_metadata = {}
        for output_name, shape in AgeGender.output_shape.items():
            expected_output_metadata[output_name] = {'dtype': 1, 'shape': list(shape)}
        rest_url = 'http://localhost:{}/v1/models/age_gender/metadata'.format(
                   ports["rest_port"])
        response = get_model_metadata_response_rest(rest_url)
        print("response", response)
        input_metadata, output_metadata = model_metadata_response(response=response)

        assert response.model_spec.name == AgeGender.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
