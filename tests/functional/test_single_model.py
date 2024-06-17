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

from constants import MODEL_SERVICE, ERROR_SHAPE, NOT_TO_BE_REPORTED_IF_SKIPPED, TARGET_DEVICE_MYRIAD, \
    TARGET_DEVICE_GPU, TARGET_DEVICE_HDDL
from config import skip_nginx_test
from conftest import devices_not_supported_for_test
from model.models_information import Resnet
from utils.grpc import create_channel, infer, get_model_metadata_request, get_model_metadata, model_metadata_response, \
    get_model_status
import logging
from utils.models_utils import ModelVersionState, ErrorCode, \
    ERROR_MESSAGE  # noqa
from utils.rest import get_predict_url, get_metadata_url, get_status_url, infer_rest, \
    get_model_metadata_response_rest, get_model_status_response_rest

logger = logging.getLogger(__name__)


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD, TARGET_DEVICE_HDDL, TARGET_DEVICE_GPU])
class TestSingleModelInference:

    def test_run_inference(self, start_server_single_model):
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

        _, ports = start_server_single_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        imgs_v1_224 = np.ones(Resnet.input_shape, Resnet.dtype)
        output = infer(imgs_v1_224, input_tensor=Resnet.input_name, grpc_stub=stub,
                       model_spec_name=Resnet.name,
                       model_spec_version=None,
                       output_tensors=[Resnet.output_name])
        logger.info("Output shape: {}".format(output[Resnet.output_name].shape))
        assert output[Resnet.output_name].shape == Resnet.output_shape, ERROR_SHAPE

    def test_get_model_metadata(self, start_server_single_model):

        _, ports = start_server_single_model

        stub = create_channel(port=ports["grpc_port"])

        expected_input_metadata = {Resnet.input_name: {'dtype': 1, 'shape': list(Resnet.input_shape)}}
        expected_output_metadata = {Resnet.output_name: {'dtype': 1, 'shape': list(Resnet.output_shape)}}
        request = get_model_metadata_request(model_name=Resnet.name)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == Resnet.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata


    def test_get_model_status(self, start_server_single_model):

        _, ports = start_server_single_model
        stub = create_channel(port=ports["grpc_port"], service=MODEL_SERVICE)
        request = get_model_status(model_name=Resnet.name)
        response = stub.GetModelStatus(request, 60)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

    @pytest.mark.parametrize("request_format",
                             ['row_name', 'row_noname', 'column_name', 'column_noname'])
    def test_run_inference_rest(self, start_server_single_model, request_format):
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

        _, ports = start_server_single_model
        imgs_v1_224 = np.ones(Resnet.input_shape, Resnet.dtype)
        rest_url = get_predict_url(model=Resnet.name, port=ports["rest_port"])
        output = infer_rest(imgs_v1_224, input_tensor=Resnet.input_name,
                            rest_url=rest_url,
                            output_tensors=[Resnet.output_name],
                            request_format=request_format)
        logger.info("Output shape: {}".format(output[Resnet.output_name].shape))
        assert output[Resnet.output_name].shape == Resnet.output_shape, ERROR_SHAPE

    def test_get_model_metadata_rest(self, start_server_single_model):

        _, ports = start_server_single_model
        expected_input_metadata = {Resnet.input_name: {'dtype': 1, 'shape': list(Resnet.input_shape)}}
        expected_output_metadata = {Resnet.output_name: {'dtype': 1, 'shape': list(Resnet.output_shape)}}
        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"])
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == Resnet.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status_rest(self, start_server_single_model):

        _, ports = start_server_single_model
        rest_url = get_status_url(model=Resnet.name, port=ports["rest_port"])
        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
