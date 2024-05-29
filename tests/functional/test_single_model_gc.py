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

from constants import MODEL_SERVICE, ERROR_SHAPE, NOT_TO_BE_REPORTED_IF_SKIPPED, TARGET_DEVICE_MYRIAD, \
    TARGET_DEVICE_HDDL, TARGET_DEVICE_CUDA
from config import skip_nginx_test
from conftest import devices_not_supported_for_test
from model.models_information import Resnet, ResnetGS
from utils.grpc import create_channel, infer, get_model_metadata_request, get_model_metadata, model_metadata_response, \
    get_model_status
import logging
from utils.models_utils import ModelVersionState, ErrorCode, ERROR_MESSAGE

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
@devices_not_supported_for_test([TARGET_DEVICE_MYRIAD, TARGET_DEVICE_HDDL, TARGET_DEVICE_CUDA])
class TestSingleModelInferenceGc:

    def test_run_inference(self, start_server_single_model_from_gc):
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
        _, ports = start_server_single_model_from_gc
        stub = create_channel(port=ports["grpc_port"])

        imgs_v1_224 = np.ones(ResnetGS.input_shape, ResnetGS.dtype)
        out_name = '1463'
        output = infer(imgs_v1_224, input_tensor='0', grpc_stub=stub,
                       model_spec_name=Resnet.name,
                       model_spec_version=None,
                       output_tensors=[out_name])
        logger.info("Output shape: ".format(output[out_name].shape))
        assert output[out_name].shape == ResnetGS.output_shape, ERROR_SHAPE

    @pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED) 
    def test_get_model_metadata(self, start_server_single_model_from_gc):

        _, ports = start_server_single_model_from_gc
        stub = create_channel(port=ports["grpc_port"])

        out_name = '1463'
        expected_input_metadata = {'0': {'dtype': 1, 'shape': list(ResnetGS.input_shape)}}
        expected_output_metadata = {out_name: {'dtype': 1, 'shape': list(ResnetGS.output_shape)}}
        request = get_model_metadata_request(model_name=Resnet.name)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert response.model_spec.name == Resnet.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    @pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED) 
    def test_get_model_status(self, start_server_single_model_from_gc):

        _, ports = start_server_single_model_from_gc
        stub = create_channel(port=ports["grpc_port"], service=MODEL_SERVICE)
        request = get_model_status(model_name=Resnet.name)
        response = stub.GetModelStatus(request, 60)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 2
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
