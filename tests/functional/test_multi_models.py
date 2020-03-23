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
from utils.grpc import infer, get_model_metadata, \
    model_metadata_response, get_model_status
from utils.rest import infer_rest, \
    get_model_metadata_response_rest, get_model_status_response_rest

sys.path.append(".")
from ie_serving.models.models_utils import ModelVersionState, ErrorCode, \
    _ERROR_MESSAGE  # noqa


class TestMultiModelInference():

    def test_run_inference(self, resnet_multiple_batch_sizes,
                           start_server_multi_model,
                           create_grpc_channel):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'

        img = np.ones((1, 3, 224, 224))
        print("Starting inference using resnet model")
        model_name = "resnet"
        output = infer(img, input_tensor=in_name,
                       grpc_stub=stub,
                       model_spec_name=model_name,
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1001), ERROR_SHAPE

        model_name = "resnet_bs4"
        imgs = np.ones((4, 3, 224, 224))
        print("Starting inference using resnet model")
        output = infer(imgs, input_tensor=in_name,
                       grpc_stub=stub,
                       model_spec_name=model_name,
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (4, 1001), ERROR_SHAPE

        model_name = "resnet_bs8"
        imgs = np.ones((8, 3, 224, 224))
        print("Starting inference using resnet model")
        output = infer(imgs, input_tensor=in_name,
                       grpc_stub=stub,
                       model_spec_name=model_name,
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (8, 1001), ERROR_SHAPE

        output = infer(img, input_tensor=in_name, grpc_stub=stub,
                       model_spec_name='resnet_s3',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1001), ERROR_SHAPE

        in_name = 'input'
        out_name = 'resnet_v1_50/predictions/Reshape_1'

        img = np.ones((1, 3, 224, 224))
        in_name = 'data'
        out_name = 'prob'
        output = infer(img, input_tensor=in_name, grpc_stub=stub,
                       model_spec_name='resnet_gs',
                       model_spec_version=None,
                       output_tensors=[out_name])
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1000), ERROR_SHAPE

    def test_get_model_metadata(self, resnet_multiple_batch_sizes,
                                start_server_multi_model,
                                create_grpc_channel):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9001', PREDICTION_SERVICE)
        out_name = 'softmax_tensor'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        print("Getting info about resnet model")
        model_name = 'resnet'

        expected_input_metadata = {in_name: {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1001]}}
        request = get_model_metadata(model_name=model_name)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

        model_name = 'resnet_bs4'
        request = get_model_metadata(model_name=model_name)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        expected_input_metadata = {in_name: {'dtype': 1,
                                             'shape': [4, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [4, 1001]}}
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status(self, resnet_multiple_batch_sizes,
                              start_server_multi_model,
                              create_grpc_channel):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        stub = create_grpc_channel('localhost:9001', MODEL_SERVICE)
        request = get_model_status(model_name='resnet', version=1)
        response = stub.GetModelStatus(request, 10)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

        request = get_model_status(model_name='resnet_bs4')
        response = stub.GetModelStatus(request, 10)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

    def test_run_inference_rest(self, resnet_multiple_batch_sizes,
                                start_server_multi_model):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        img = np.ones((1, 3, 224, 224))
        print("Starting inference using resnet model")
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'

        model_name = 'resnet'
        rest_url = 'http://localhost:5561/v1/models/{}:predict'.format(
            model_name)
        output = infer_rest(img, input_tensor=in_name, rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format='column_name')
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1001), ERROR_SHAPE

        imgs = np.ones((4, 3, 224, 224))
        model_name = 'resnet_bs4'
        rest_url = 'http://localhost:5561/v1/models/{}:predict'.format(
            model_name)
        output = infer_rest(imgs, input_tensor=in_name, rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format='row_noname')
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (4, 1001), ERROR_SHAPE

        imgs = np.ones((8, 3, 224, 224))
        model_name = 'resnet_bs8'
        rest_url = 'http://localhost:5561/v1/models/{}:predict'.format(
            model_name)
        output = infer_rest(imgs, input_tensor=in_name, rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format='row_noname')
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (8, 1001), ERROR_SHAPE

        model_name = 'resnet_s3'
        rest_url = 'http://localhost:5561/v1/models/{}:predict'.format(
            model_name)
        output = infer_rest(img, input_tensor=in_name, rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format='row_name')
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1001), ERROR_SHAPE

        in_name = 'input'
        out_name = 'resnet_v1_50/predictions/Reshape_1'

        model_name = 'resnet_gs'
        rest_url = 'http://localhost:5561/v1/models/{}:predict'.format(
            model_name)
        output = infer_rest(img, input_tensor=in_name, rest_url=rest_url,
                            output_tensors=[out_name],
                            request_format='column_noname')
        print("output shape", output[out_name].shape)
        assert output[out_name].shape == (1, 1000), ERROR_SHAPE

    def test_get_model_metadata_rest(self, resnet_multiple_batch_sizes,
                                     start_server_multi_model):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        print("Getting info about resnet model")
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        model_name = 'resnet'
        expected_input_metadata = {in_name: {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1001]}}
        rest_url = 'http://localhost:5561/v1/models/{}/metadata'.format(
            model_name)
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

        model_name = 'resnet_bs4'
        rest_url = 'http://localhost:5561/v1/models/{}/metadata'.format(
            model_name)
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        expected_input_metadata = {in_name: {'dtype': 1,
                                             'shape': [4, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [4, 1001]}}
        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

    def test_get_model_status_rest(self, resnet_multiple_batch_sizes,
                                   start_server_multi_model):
        print("Downloaded model files:", resnet_multiple_batch_sizes)

        rest_url = 'http://localhost:5561/v1/models/resnet'
        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]

        rest_url = 'http://localhost:5561/v1/models/resnet_bs4/versions/1'
        response = get_model_status_response_rest(rest_url)
        versions_statuses = response.model_version_status
        version_status = versions_statuses[0]
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
