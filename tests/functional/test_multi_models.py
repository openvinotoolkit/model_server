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
from constants import PREDICTION_SERVICE, MODEL_SERVICE, ERROR_SHAPE
from utils.grpc import infer, infer_batch, get_model_metadata, \
    model_metadata_response, get_model_status
from utils.rest import infer_batch_rest, infer_rest, \
    get_model_metadata_response_rest, get_model_status_response_rest

sys.path.append(".")
from ie_serving.models.models_utils import ModelVersionState, ErrorCode, \
    _ERROR_MESSAGE  # noqa


class TestMuiltModelInference():

    def test_run_inference(self, download_two_models,
                           input_data_downloader_v1_224,
                           input_data_downloader_v3_331,
                           start_server_multi_model,
                           create_grpc_channel):
        """
        <b>Description</b>
        Execute inference request using gRPC interface hosting multiple models

        <b>input data</b>
        - directory with 2 models in IR format
        - docker image
        - input data in numpy format

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape for both models set in config
        file: model resnet_v1_50, pnasnet_large
        - both served models handles appropriate input formats

        """

        print("Downloaded model files:", download_two_models)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)

        input_data = input_data_downloader_v1_224[:2, :, :, :]
        print("Starting inference using resnet model")
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer_batch(input_data, input_tensor='input',
                                 grpc_stub=stub,
                                 model_spec_name='resnet_V1_50',
                                 model_spec_version=None,
                                 output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (2, 1000), ERROR_SHAPE

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet_gs',
                           model_spec_version=None,
                           output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (1, 1000), ERROR_SHAPE

        out_name = 'resnet_v1_50/predictions/Reshape_1'
        for x in range(0, 10):
            output = infer(imgs_v1_224, slice_number=x,
                           input_tensor='input', grpc_stub=stub,
                           model_spec_name='resnet_s3',
                           model_spec_version=None,
                           output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (1, 1000), ERROR_SHAPE

        input_data = input_data_downloader_v3_331[:4, :, :, :]
        print("Starting inference using pnasnet_large model")
        out_name = 'final_layer/predictions'
        for x in range(0, 10):
            output = infer_batch(input_data, input_tensor='input',
                                 grpc_stub=stub,
                                 model_spec_name='pnasnet_large',
                                 model_spec_version=None,
                                 output_tensors=[out_name])
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (4, 1001), ERROR_SHAPE

    def test_get_model_metadata(self, download_two_models,
                                start_server_multi_model,
                                create_grpc_channel):
        """
        <b>Description</b>
        Execute inference request using gRPC interface hosting multiple models

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
        print("Downloaded model files:", download_two_models)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)

        print("Getting info about resnet model")
        model_name = 'resnet_V1_50'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [2, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [2, 1000]}}
        request = get_model_metadata(model_name='resnet_V1_50')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

        model_name = 'pnasnet_large'
        out_name = 'final_layer/predictions'
        request = get_model_metadata(model_name='pnasnet_large')
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [4, 3, 331, 331]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [4, 1001]}}
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status(self, download_two_models,
                              start_server_multi_model,
                              create_grpc_channel):

        print("Downloaded model files:", download_two_models)

        stub = create_grpc_channel('localhost:9001', MODEL_SERVICE)
        request = get_model_status(model_name='resnet_V1_50', version=1)
        response = stub.GetModelStatus(request, 10)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

        request = get_model_status(model_name='pnasnet_large')
        response = stub.GetModelStatus(request, 10)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

    def test_run_inference_rest(self, download_two_models,
                                input_data_downloader_v1_224,
                                input_data_downloader_v3_331,
                                start_server_multi_model):
        """
        <b>Description</b>
        Execute inference request using REST API interface hosting multiple
        models

        <b>input data</b>
        - directory with 2 models in IR format
        - docker image
        - input data in numpy format

        <b>fixtures used</b>
        - model downloader
        - input data downloader
        - service launching

        <b>Expected results</b>
        - response contains proper numpy shape for both models set in config
        file: model resnet_v1_50, pnasnet_large
        - both served models handles appropriate input formats

        """

        print("Downloaded model files:", download_two_models)

        input_data = input_data_downloader_v1_224[:2, :, :, :]
        print("Starting inference using resnet model")
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        rest_url = 'http://localhost:5561/v1/models/resnet_V1_50:predict'
        for x in range(0, 10):
            output = infer_batch_rest(input_data,
                                      input_tensor='input', rest_url=rest_url,
                                      output_tensors=[out_name],
                                      request_format='column_name')
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (2, 1000), ERROR_SHAPE

        imgs_v1_224 = np.array(input_data_downloader_v1_224)
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        rest_url = 'http://localhost:5561/v1/models/resnet_gs:predict'
        for x in range(0, 10):
            output = infer_rest(imgs_v1_224, slice_number=x,
                                input_tensor='input', rest_url=rest_url,
                                output_tensors=[out_name],
                                request_format='column_noname')
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (1, 1000), ERROR_SHAPE

        out_name = 'resnet_v1_50/predictions/Reshape_1'
        rest_url = 'http://localhost:5561/v1/models/resnet_s3:predict'
        for x in range(0, 10):
            output = infer_rest(imgs_v1_224, slice_number=x,
                                input_tensor='input', rest_url=rest_url,
                                output_tensors=[out_name],
                                request_format='row_name')
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (1, 1000), ERROR_SHAPE

        input_data = input_data_downloader_v3_331[:4, :, :, :]
        print("Starting inference using pnasnet_large model")
        out_name = 'final_layer/predictions'
        rest_url = 'http://localhost:5561/v1/models/pnasnet_large:predict'
        for x in range(0, 10):
            output = infer_batch_rest(input_data,
                                      input_tensor='input', rest_url=rest_url,
                                      output_tensors=[out_name],
                                      request_format='row_noname')
            print("output shape", output[out_name].shape)
            assert output[out_name].shape == (4, 1001), ERROR_SHAPE

    def test_get_model_metadata_rest(self, download_two_models,
                                     start_server_multi_model):
        """
        <b>Description</b>
        Execute inference request using REST API interface hosting multiple
        models

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
        print("Downloaded model files:", download_two_models)

        print("Getting info about resnet model")
        model_name = 'resnet_V1_50'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [2, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [2, 1000]}}
        rest_url = 'http://localhost:5561/v1/models/resnet_V1_50/metadata'
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

        model_name = 'pnasnet_large'
        out_name = 'final_layer/predictions'
        rest_url = 'http://localhost:5561/v1/models/pnasnet_large/metadata'
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [4, 3, 331, 331]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [4, 1001]}}
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status_rest(self, download_two_models,
                                   start_server_multi_model):

        print("Downloaded model files:", download_two_models)

        rest_url = 'http://localhost:5561/v1/models/resnet_V1_50'
        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

        rest_url = 'http://localhost:5561/v1/models/pnasnet_large/versions/1'
        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
