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
from tests.functional.constants.constants import ERROR_SHAPE, TARGET_DEVICE_MYRIAD, NOT_TO_BE_REPORTED_IF_SKIPPED
from tests.functional.config import skip_nginx_test
from tests.functional.conftest import devices_not_supported_for_test
from tests.functional.model.models_information import AgeGender
from tests.functional.utils.grpc import create_channel, infer, get_model_metadata_request, get_model_metadata, \
    model_metadata_response
import logging
from tests.functional.utils.rest import get_predict_url, get_metadata_url, infer_rest, get_model_metadata_response_rest

logger = logging.getLogger(__name__)


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD])
class TestSingleModelMappingInference:

    @pytest.mark.api_enabling
    def test_run_inference(self, start_server_with_mapping):
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

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        imgs_v1_224 = np.ones(AgeGender.input_shape, AgeGender.dtype)

        output = infer(imgs_v1_224, input_tensor=AgeGender.input_name, grpc_stub=stub,
                       model_spec_name=AgeGender.name,
                       model_spec_version=None,
                       output_tensors=AgeGender.output_name)
        for output_name, shape in AgeGender.output_shape.items():
            logger.info("Output shape: {}".format(output[output_name].shape))
            assert output[output_name].shape == shape, ERROR_SHAPE

    @pytest.mark.api_enabling
    def test_get_model_metadata(self, start_server_with_mapping):

        _, ports = start_server_with_mapping

        stub = create_channel(port=ports["grpc_port"])

        expected_input_metadata = {AgeGender.input_name: {'dtype': 1, 'shape': list(AgeGender.input_shape)}}
        expected_output_metadata = {}
        for output_name, shape in AgeGender.output_shape.items():
            expected_output_metadata[output_name] = {'dtype': 1, 'shape': list(shape)}
        request = get_model_metadata_request(model_name=AgeGender.name)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == AgeGender.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname',
                              'column_name', 'column_noname'])
    @pytest.mark.api_enabling
    def test_run_inference_rest(self, start_server_with_mapping, request_format):
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

        imgs_v1_224 = np.ones(AgeGender.input_shape, AgeGender.dtype)
        rest_url = get_predict_url(model=AgeGender.name, port=ports["rest_port"])
        output = infer_rest(imgs_v1_224, input_tensor=AgeGender.input_name,
                            rest_url=rest_url,
                            output_tensors=AgeGender.output_name,
                            request_format=request_format)
        logger.info("Output: {}".format(output))
        for output_name, shape in AgeGender.output_shape.items():
            logger.info("Output shape: {}".format(output[output_name].shape))
            assert output[output_name].shape == shape, ERROR_SHAPE

    @pytest.mark.api_enabling
    def test_get_model_metadata_rest(self, start_server_with_mapping):

        _, ports = start_server_with_mapping

        expected_input_metadata = {AgeGender.input_name: {'dtype': 1, 'shape': list(AgeGender.input_shape)}}
        expected_output_metadata = {}
        for output_name, shape in AgeGender.output_shape.items():
            expected_output_metadata[output_name] = {'dtype': 1, 'shape': list(shape)}
        rest_url = get_metadata_url(model=AgeGender.name, port=ports["rest_port"])
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == AgeGender.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
