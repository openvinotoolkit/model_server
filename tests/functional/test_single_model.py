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
import sys

import numpy as np
import pytest
from constants import MODEL_SERVICE, PREDICTION_SERVICE, ERROR_SHAPE
from utils.grpc import infer, get_model_metadata, model_metadata_response, \
    get_model_status
from utils.rest import infer_rest, get_model_metadata_response_rest, \
    get_model_status_response_rest

sys.path.append(".")
from ie_serving.models.models_utils import ModelVersionState, ErrorCode, \
    _ERROR_MESSAGE  # noqa


class TestSingleModelInference():

    def test_run_inference(self, resnet_v1_50_model_downloader,
                           input_data_downloader_v1_224,
                           start_server_single_model,
                           create_grpc_channel):
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

        print("Downloaded model files:", resnet_v1_50_model_downloader)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9000', PREDICTION_SERVICE)

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet',
                           model_spec_version=None,
                           output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1000), ERROR_SHAPE

    def test_get_model_metadata(self, resnet_v1_50_model_downloader,
                                start_server_single_model,
                                create_grpc_channel):

        print("Downloaded model files:", resnet_v1_50_model_downloader)

        stub = create_grpc_channel('localhost:9000', PREDICTION_SERVICE)

        model_name = 'resnet'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1000]}}
        request = get_model_metadata(model_name='resnet')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status(self, resnet_v1_50_model_downloader,
                              start_server_single_model,
                              create_grpc_channel):

        print("Downloaded model files:", resnet_v1_50_model_downloader)

        stub = create_grpc_channel('localhost:9000', MODEL_SERVICE)
        request = get_model_status(model_name='resnet')
        response = stub.GetModelStatus(request, 10)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

    @pytest.mark.parametrize("request_format",
                             [('row_name'), ('row_noname'),
                              ('column_name'), ('column_noname')])
    def test_run_inference_rest(self, resnet_v1_50_model_downloader,
                                input_data_downloader_v1_224,
                                start_server_single_model,
                                request_format):
        """
        <b>Description</b>
        Submit request to REST API interface serving a single resnet model

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

        print("Downloaded model files:", resnet_v1_50_model_downloader)

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        rest_url = 'http://localhost:5555/v1/models/resnet:predict'
        for x in range(0, 10):
            output = infer_rest(imgs_v1_224, slice_number=x,
                                input_tensor='input', rest_url=rest_url,
                                output_tensors=[out_name],
                                request_format=request_format)
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (1, 1000), ERROR_SHAPE

    def test_get_model_metadata_rest(self, resnet_v1_50_model_downloader,
                                     start_server_single_model):

        print("Downloaded model files:", resnet_v1_50_model_downloader)

        model_name = 'resnet'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1000]}}
        rest_url = 'http://localhost:5555/v1/models/resnet/metadata'
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status_rest(self, resnet_v1_50_model_downloader,
                                   start_server_single_model):

        print("Downloaded model files:", resnet_v1_50_model_downloader)

        rest_url = 'http://localhost:5555/v1/models/resnet'
        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
