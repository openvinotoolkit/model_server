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
from tests.functional.constants.constants import MODEL_SERVICE, ERROR_SHAPE, TARGET_DEVICE_GPU, TARGET_DEVICE_HDDL, \
    NOT_TO_BE_REPORTED_IF_SKIPPED, TARGET_DEVICE_MYRIAD
from tests.functional.config import  skip_nginx_test
from tests.functional.conftest import devices_not_supported_for_test
from tests.functional.model.models_information import Resnet, ResnetBS4, ResnetBS8, ResnetS3
from tests.functional.utils.grpc import create_channel, infer, get_model_metadata_request, get_model_metadata, \
    model_metadata_response, get_model_status
import logging
from tests.functional.utils.models_utils import ModelVersionState, ErrorCode, \
    ERROR_MESSAGE  # noqa
from tests.functional.utils.rest import get_predict_url, get_metadata_url, get_status_url, infer_rest, \
    get_model_metadata_response_rest, get_model_status_response_rest

logger = logging.getLogger(__name__)


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD, TARGET_DEVICE_HDDL, TARGET_DEVICE_GPU])
class TestMultiModelInference:

    @pytest.mark.api_enabling
    def test_run_inference(self, start_server_multi_model):

        _, ports = start_server_multi_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        for model in [Resnet, ResnetBS4, ResnetBS8, ResnetS3]:
            input_data = np.ones(model.input_shape, model.dtype)
            logger.info("Starting inference using {} model".format(model.name))
            output = infer(input_data, input_tensor=model.input_name,
                           grpc_stub=stub,
                           model_spec_name=model.name,
                           model_spec_version=None,
                           output_tensors=[model.output_name])
            logger.info("Output shape: {} for model {} ".format(output[model.output_name].shape, model.name))
            assert_msg = "{} for model {}".format(ERROR_SHAPE, model.name)
            assert output[model.output_name].shape == model.output_shape, assert_msg

    @pytest.mark.api_enabling
    def test_get_model_metadata(self, start_server_multi_model):
        _, ports = start_server_multi_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        for model in [Resnet, ResnetBS4, ResnetBS8, ResnetS3]:
            logger.info("Getting info about {} model".format(model.name))
            expected_input_metadata = {model.input_name: {'dtype': 1, 'shape': list(model.input_shape)}}
            expected_output_metadata = {model.output_name: {'dtype': 1, 'shape': list(model.output_shape)}}
            request = get_model_metadata_request(model_name=model.name)
            response = get_model_metadata(stub, request)
            input_metadata, output_metadata = model_metadata_response(response=response)
            logger.info("Input metadata: {}".format(input_metadata))
            logger.info("Output metadata: {}".format(output_metadata))

            assert response.model_spec.name == model.name
            assert expected_input_metadata == input_metadata
            assert expected_output_metadata == output_metadata

    @pytest.mark.api_enabling
    def test_get_model_status(self, start_server_multi_model):

        _, ports = start_server_multi_model

        stub = create_channel(port=ports["grpc_port"], service=MODEL_SERVICE)

        for model in [Resnet, ResnetBS4, ResnetBS8, ResnetS3]:
            request = get_model_status(model_name=model.name, version=1)
            response = stub.GetModelStatus(request, 60)
            versions_statuses = response.model_version_status
            version_status = versions_statuses[0]
            assert version_status.version == 1
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]

    @pytest.mark.api_enabling
    def test_run_inference_rest(self, start_server_multi_model):

        _, ports = start_server_multi_model

        for model in [Resnet, ResnetBS4, ResnetBS8, ResnetS3]:
            input_data = np.ones(model.input_shape, model.dtype)
            logger.info("Starting inference using {} model".format(model.name))

            rest_url = get_predict_url(model=model.name, port=ports["rest_port"])
            output = infer_rest(input_data, input_tensor=model.input_name, rest_url=rest_url,
                                output_tensors=[model.output_name],
                                request_format=model.rest_request_format)
            logger.info("Output shape: {}".format(output[model.output_name].shape))
            assert output[model.output_name].shape == model.output_shape, ERROR_SHAPE

    @pytest.mark.api_enabling
    def test_get_model_metadata_rest(self, start_server_multi_model):

        _, ports = start_server_multi_model

        for model in [Resnet, ResnetBS4]:
            logger.info("Getting info about {} model".format(model.name))
            expected_input_metadata = {model.input_name: {'dtype': 1, 'shape': list(model.input_shape)}}
            expected_output_metadata = {model.output_name: {'dtype': 1, 'shape': list(model.output_shape)}}
            rest_url = get_metadata_url(model=model.name, port=ports["rest_port"])
            response = get_model_metadata_response_rest(rest_url)
            input_metadata, output_metadata = model_metadata_response(response=response)
            logger.info("Input metadata: {}".format(input_metadata))
            logger.info("Output metadata: {}".format(output_metadata))

            assert response.model_spec.name == model.name
            assert expected_input_metadata == input_metadata
            assert expected_output_metadata == output_metadata

    @pytest.mark.api_enabling
    def test_get_model_status_rest(self, start_server_multi_model):

        _, ports = start_server_multi_model

        for model in [Resnet, ResnetBS4]:
            rest_url = get_status_url(model=model.name, port=ports["rest_port"])
            response = get_model_status_response_rest(rest_url)
            versions_statuses = response.model_version_status
            version_status = versions_statuses[0]
            assert version_status.version == 1
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
