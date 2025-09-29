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

from tests.functional.config import skip_nginx_test
from tests.functional.constants.constants import MODEL_SERVICE, TARGET_DEVICE_GPU, TARGET_DEVICE_HDDL, \
    TARGET_DEVICE_MYRIAD, TARGET_DEVICE_CUDA, NOT_TO_BE_REPORTED_IF_SKIPPED
from tests.functional.conftest import devices_not_supported_for_test
from tests.functional.model.models_information import PVBFaceDetectionV2, PVBFaceDetection
from tests.functional.utils.grpc import create_channel, infer, get_model_metadata_request, get_model_metadata, model_metadata_response, \
    get_model_status
import logging
from tests.functional.utils.models_utils import ModelVersionState, ErrorCode, \
    ERROR_MESSAGE  # noqa
from tests.functional.utils.rest import get_predict_url, get_metadata_url, get_status_url, infer_rest, \
    get_model_metadata_response_rest, get_model_status_response_rest

logger = logging.getLogger(__name__)


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD, TARGET_DEVICE_HDDL, TARGET_DEVICE_GPU, TARGET_DEVICE_CUDA])
class TestModelVersionHandling:
    model_name = "pvb_face_multi_version"

    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    @pytest.mark.api_enabling
    def test_run_inference(self, start_server_multi_model, version):

        _, ports = start_server_multi_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])
        model_info = PVBFaceDetectionV2 if version is None else PVBFaceDetection[version - 1]

        img = np.ones(model_info.input_shape, dtype=model_info.dtype)

        output = infer(img, input_tensor=model_info.input_name,
                       grpc_stub=stub, model_spec_name=self.model_name,
                       model_spec_version=version,  # face detection
                       output_tensors=[model_info.output_name])
        logger.info("Output shape: {}".format(output[model_info.output_name].shape))
        assert output[model_info.output_name].shape == model_info.output_shape, \
            '{} with version 1 has invalid output'.format(self.model_name)

    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    @pytest.mark.api_enabling
    def test_get_model_metadata(self, start_server_multi_model, version):

        _, ports = start_server_multi_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"])
        model_info = PVBFaceDetectionV2 if version is None else PVBFaceDetection[version - 1]

        logger.info("Getting info about pvb_face_detection model "
              "version: {}".format("no_version" if version is None else version))
        expected_input_metadata = {model_info.input_name: {'dtype': 1, 'shape': list(model_info.input_shape)}}
        expected_output_metadata = {model_info.output_name: {'dtype': 1, 'shape': list(model_info.output_shape)}}

        request = get_model_metadata_request(model_name=self.model_name,
                                     version=version)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == self.model_name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    @pytest.mark.api_enabling
    def test_get_model_status(self, start_server_multi_model, version):

        _, ports = start_server_multi_model

        # Connect to grpc service
        stub = create_channel(port=ports["grpc_port"], service=MODEL_SERVICE)
        request = get_model_status(model_name=self.model_name,
                                   version=version)
        response = stub.GetModelStatus(request, 60)

        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        if version is None:
            assert len(versions_statuses) == 2
        else:
            assert version_status.version == version
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    @pytest.mark.api_enabling
    def test_run_inference_rest(self, start_server_multi_model, version):

        _, ports = start_server_multi_model

        model_info = PVBFaceDetectionV2 if version is None else PVBFaceDetection[version - 1]

        img = np.ones(model_info.input_shape, dtype=model_info.dtype)
        rest_url = get_predict_url(model=self.model_name, port=ports["rest_port"], version=version)
        output = infer_rest(img,
                            input_tensor=model_info.input_name, rest_url=rest_url,
                            output_tensors=[model_info.output_name],
                            request_format='column_name')
        logger.info("Output shape: {}".format(output[model_info.output_name].shape))
        assert output[model_info.output_name].shape == model_info.output_shape, \
            '{} with version 1 has invalid output'.format(self.model_name)

    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    @pytest.mark.api_enabling
    def test_get_model_metadata_rest(self, start_server_multi_model, version):

        _, ports = start_server_multi_model

        model_info = PVBFaceDetectionV2 if version is None else PVBFaceDetection[version - 1]

        rest_url = get_metadata_url(model=self.model_name, port=ports["rest_port"], version=version)

        expected_input_metadata = {model_info.input_name: {'dtype': 1, 'shape': list(model_info.input_shape)}}
        expected_output_metadata = {model_info.output_name: {'dtype': 1, 'shape': list(model_info.output_shape)}}
        logger.info("Getting info about resnet model version: {}".format(rest_url))
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == self.model_name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.parametrize("version", [1, 2, None], ids=("version 1", "version 2", "no version specified"))
    @pytest.mark.api_enabling
    def test_get_model_status_rest(self, start_server_multi_model, version):

        _, ports = start_server_multi_model

        rest_url = get_status_url(model=self.model_name, port=ports["rest_port"], version=version)

        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        if version is None:
            assert len(versions_statuses) == 2
        else:
            assert version_status.version == version
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
