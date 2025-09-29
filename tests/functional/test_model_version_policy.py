#
# Copyright (c) 2019 Intel Corporation
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
import requests
from google.protobuf.json_format import Parse
from tensorflow_serving.apis import get_model_metadata_pb2, \
    get_model_status_pb2  # noqa

from tests.functional.constants.constants import MODEL_SERVICE, TARGET_DEVICE_MYRIAD, TARGET_DEVICE_CUDA, \
    NOT_TO_BE_REPORTED_IF_SKIPPED
from tests.functional.config import skip_nginx_test
from tests.functional.conftest import devices_not_supported_for_test
from tests.functional.model.models_information import AgeGender, PVBDetection, PVBFaceDetectionV2
from tests.functional.utils.grpc import create_channel, get_model_metadata_request, get_model_metadata, \
    model_metadata_response, get_model_status
import logging
from tests.functional.utils.models_utils import ModelVersionState, ErrorCode, \
    ERROR_MESSAGE  # noqa
from tests.functional.utils.rest import get_metadata_url, get_status_url, get_model_status_response_rest

logger = logging.getLogger(__name__)


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD, TARGET_DEVICE_CUDA])
class TestModelVerPolicy:

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
    @pytest.mark.api_enabling
    def test_get_model_metadata(self, model_version_policy_models,
                                start_server_model_ver_policy,
                                model_name, throw_error):

        _, ports = start_server_model_ver_policy
        logger.info("Downloaded model files: {}".format(model_version_policy_models))

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])

        versions = [1, 2, 3]
        expected_outputs_metadata = [
            {PVBDetection.output_name: {'dtype': 1, 'shape': list(PVBDetection.output_shape)}},
            {PVBFaceDetectionV2.output_name: {'dtype': 1, 'shape': list(PVBFaceDetectionV2.output_shape)}}]
        expected_output_metadata = {}
        for output_name, shape in AgeGender.output_shape.items():
            expected_output_metadata[output_name] = {'dtype': 1, 'shape': list(shape)}
            expected_outputs_metadata.append(expected_output_metadata)
        expected_inputs_metadata = [
            {PVBDetection.input_name: {'dtype': 1, 'shape': list(PVBDetection.input_shape)}},
            {PVBFaceDetectionV2.input_name: {'dtype': 1, 'shape': list(PVBFaceDetectionV2.input_shape)}},
            {AgeGender.input_name: {'dtype': 1, 'shape': list(AgeGender.input_shape)}}]

        for x in range(len(versions)):
            logger.info("Getting info about model version: {}".format(versions[x]))
            expected_input_metadata = expected_inputs_metadata[x]
            expected_output_metadata = expected_outputs_metadata[x]
            request = get_model_metadata_request(model_name=model_name, 
                                                 version=versions[x])
            if not throw_error[x]:
                response = get_model_metadata(stub, request)
                input_metadata, output_metadata = model_metadata_response(
                    response=response)

                logger.info("Input metadata: {}".format(input_metadata))
                logger.info("Output metadata: {}".format(output_metadata))

                assert model_name == response.model_spec.name
                assert expected_input_metadata == input_metadata
                assert expected_output_metadata == output_metadata
            else:
                with pytest.raises(Exception) as e:
                    get_model_metadata(stub, request)
                assert "Model with requested version is not found" in str(e.value)

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
    @pytest.mark.api_enabling
    def test_get_model_status(self, model_version_policy_models,
                              start_server_model_ver_policy,
                              model_name, throw_error):

        _, ports = start_server_model_ver_policy
        logger.info("Downloaded model files: {}".format(model_version_policy_models))

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"], service=MODEL_SERVICE)

        versions = [1, 2, 3]
        for x in range(len(versions)):
            request = get_model_status(model_name=model_name,
                                       version=versions[x])
            if not throw_error[x]:
                response = stub.GetModelStatus(request, 60)
                versions_statuses = response.model_version_status
                version_status = versions_statuses[0]
                assert version_status.version == versions[x]
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
            else:
                with pytest.raises(Exception) as e:
                    stub.GetModelStatus(request, 60)
                assert "Model with requested version is not found" in str(e.value)

        #   aggregated results check
        if model_name == 'all':
            request = get_model_status(model_name=model_name)
            response = stub.GetModelStatus(request, 60)
            versions_statuses = response.model_version_status
            assert len(versions_statuses) == 3
            for version_status in versions_statuses:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
    @pytest.mark.api_enabling
    def test_get_model_metadata_rest(self, model_version_policy_models,
                                     start_server_model_ver_policy,
                                     model_name, throw_error):
        """
        <b>Description</b>
        Execute GetModelMetadata request using REST API interface
        hosting multiple models

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

        _, ports = start_server_model_ver_policy
        logger.info("Downloaded model files: {}".format(model_version_policy_models))

        logger.info("Getting info about models")
        versions = [1, 2, 3]
        expected_outputs_metadata = [
            {PVBDetection.output_name: {'dtype': 1, 'shape': list(PVBDetection.output_shape)}},
            {PVBFaceDetectionV2.output_name: {'dtype': 1, 'shape': list(PVBFaceDetectionV2.output_shape)}}]
        expected_output_metadata = {}
        for output_name, shape in AgeGender.output_shape.items():
            expected_output_metadata[output_name] = {'dtype': 1, 'shape': list(shape)}
            expected_outputs_metadata.append(expected_output_metadata)
        expected_inputs_metadata = [
            {PVBDetection.input_name: {'dtype': 1, 'shape': list(PVBDetection.input_shape)}},
            {PVBFaceDetectionV2.input_name: {'dtype': 1, 'shape': list(PVBFaceDetectionV2.input_shape)}},
            {AgeGender.input_name: {'dtype': 1, 'shape': list(AgeGender.input_shape)}}]

        for x in range(len(versions)):
            logger.info("Getting info about model version: {}".format(versions[x]))
            expected_input_metadata = expected_inputs_metadata[x]
            expected_output_metadata = expected_outputs_metadata[x]
            rest_url = get_metadata_url(model=model_name, port=ports["rest_port"], version=str(versions[x]))
            result = requests.get(rest_url)
            logger.info("Result: {}".format(result.text))
            if not throw_error[x]:
                output_json = result.text
                metadata_pb = get_model_metadata_pb2. \
                    GetModelMetadataResponse()
                response = Parse(output_json, metadata_pb,
                                 ignore_unknown_fields=True)
                input_metadata, output_metadata = model_metadata_response(
                    response=response)

                logger.info("Input metadata: {}".format(input_metadata))
                logger.info("Output metadata: {}".format(output_metadata))

                assert model_name == response.model_spec.name
                assert expected_input_metadata == input_metadata
                assert expected_output_metadata == output_metadata
            else:
                assert 404 == result.status_code

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
    @pytest.mark.api_enabling
    def test_get_model_status_rest(self, model_version_policy_models,
                                   start_server_model_ver_policy,
                                   model_name, throw_error):

        _, ports = start_server_model_ver_policy
        logger.info("Downloaded model files: {}".format(model_version_policy_models))

        versions = [1, 2, 3]
        for x in range(len(versions)):
            rest_url = get_status_url(model=model_name, port=ports["rest_port"], version=str(versions[x]))
            result = requests.get(rest_url)
            if not throw_error[x]:
                output_json = result.text
                status_pb = get_model_status_pb2.GetModelStatusResponse()
                response = Parse(output_json, status_pb,
                                 ignore_unknown_fields=False)
                versions_statuses = response.model_version_status
                version_status = versions_statuses[0]
                assert version_status.version == versions[x]
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
            else:
                assert 404 == result.status_code

                #   aggregated results check
        if model_name == 'all':
            rest_url = get_status_url(model=model_name, port=ports["rest_port"])
            response = get_model_status_response_rest(rest_url)
            versions_statuses = response.model_version_status
            assert len(versions_statuses) == 3
            for version_status in versions_statuses:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
