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
from tests.functional.constants.constants import MODEL_SERVICE, ERROR_SHAPE, TARGET_DEVICE_MYRIAD, TARGET_DEVICE_CUDA, \
    NOT_TO_BE_REPORTED_IF_SKIPPED
from tests.functional.config import skip_nginx_test
from tests.functional.conftest import devices_not_supported_for_test
from tests.functional.model.models_information import Resnet
from tests.functional.utils.grpc import create_channel, infer, get_model_metadata_request, get_model_metadata, \
    model_metadata_response, get_model_status
import logging
from tests.functional.utils.models_utils import ModelVersionState, ErrorCode, ERROR_MESSAGE

logger = logging.getLogger(__name__)


@pytest.mark.skipif(skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD, TARGET_DEVICE_CUDA])
class TestSingleModelInferenceS3:

    @pytest.mark.api_enabling
    def test_run_inference(self, start_server_single_model_from_minio):
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

        # Connect to grpc service
        _, ports = start_server_single_model_from_minio
        stub = create_channel(port=ports["grpc_port"])

        imgs_v1_224 = np.ones(Resnet.input_shape, Resnet.dtype)
        output = infer(imgs_v1_224, input_tensor=Resnet.input_name, grpc_stub=stub,
                       model_spec_name=Resnet.name,
                       model_spec_version=None,
                       output_tensors=[Resnet.output_name])
        logger.info("Output shape: {}".format(output[Resnet.output_name].shape))
        assert output[Resnet.output_name].shape == Resnet.output_shape, ERROR_SHAPE

    @pytest.mark.api_enabling
    def test_get_model_metadata(self, start_server_single_model_from_minio):

        _, ports = start_server_single_model_from_minio
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

    @pytest.mark.api_enabling
    def test_get_model_status(self, start_server_single_model_from_minio):

        _, ports = start_server_single_model_from_minio
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
