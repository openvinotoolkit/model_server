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
import sys

import pytest
import requests
from constants import PREDICTION_SERVICE, MODEL_SERVICE
from google.protobuf.json_format import Parse
from tensorflow_serving.apis import get_model_metadata_pb2, \
    get_model_status_pb2  # noqa
from utils.grpc import get_model_metadata, model_metadata_response, \
    get_model_status
from utils.rest import get_model_status_response_rest

sys.path.append(".")
from ie_serving.models.models_utils import ModelVersionState, ErrorCode, \
    _ERROR_MESSAGE  # noqa


class TestModelVerPolicy():

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
    def test_get_model_metadata(self, model_version_policy_models,
                                start_server_model_ver_policy,
                                create_grpc_channel,
                                model_name, throw_error):
        print("Downloaded model files:", model_version_policy_models)
        # Connect to grpc service
        stub = create_grpc_channel('localhost:9006', PREDICTION_SERVICE)
        versions = [1, 2, 3]
        expected_outputs_metadata = [
            {'detection_out': {'dtype': 1, 'shape': [1, 1, 200, 7]}},
            {'detection_out': {'dtype': 1, 'shape': [1, 1, 200, 7]}},
            {'age': {'dtype': 1, 'shape': [1, 1, 1, 1]},
             'gender': {'dtype': 1, 'shape': [1, 2, 1, 1]}}]
        expected_inputs_metadata = [
            {'data': {'dtype': 1, 'shape': [1, 3, 300, 300]}},
            {'data': {'dtype': 1, 'shape': [1, 3, 1024, 1024]}},
            {'new_key': {'dtype': 1, 'shape': [1, 3, 62, 62]}}]
        for x in range(len(versions)):
            print("Getting info about model version:".format(
                versions[x]))
            expected_input_metadata = expected_inputs_metadata[x]
            expected_output_metadata = expected_outputs_metadata[x]
            request = get_model_metadata(model_name=model_name,
                                         version=versions[x])
            if not throw_error[x]:
                response = stub.GetModelMetadata(request, 10)
                input_metadata, output_metadata = model_metadata_response(
                    response=response)

                print(output_metadata)
                assert model_name == response.model_spec.name
                assert expected_input_metadata == input_metadata
                assert expected_output_metadata == output_metadata
            else:
                with pytest.raises(Exception) as e:
                    response = stub.GetModelMetadata(request, 10)
                assert "Servable not found for request" in str(e.value)

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
    def test_get_model_status(self, model_version_policy_models,
                              start_server_model_ver_policy,
                              create_grpc_channel,
                              model_name, throw_error):

        print("Downloaded model files:", model_version_policy_models)

        # Connect to grpc service
        stub = create_grpc_channel('localhost:9006', MODEL_SERVICE)

        versions = [1, 2, 3]
        for x in range(len(versions)):
            request = get_model_status(model_name=model_name,
                                       version=versions[x])
            if not throw_error[x]:
                response = stub.GetModelStatus(request, 10)
                versions_statuses = response.model_version_status
                version_status = versions_statuses[0]
                assert version_status.version == versions[x]
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
            else:
                with pytest.raises(Exception) as e:
                    response = stub.GetModelStatus(request, 10)
                assert "Servable not found for request" in str(e.value)

        #   aggregated results check
        if model_name == 'all':
            request = get_model_status(model_name=model_name)
            response = stub.GetModelStatus(request, 10)
            versions_statuses = response.model_version_status
            assert len(versions_statuses) == 3
            for version_status in versions_statuses:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
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
        print("Downloaded model files:", model_version_policy_models)

        print("Getting info about model")
        versions = [1, 2, 3]
        expected_outputs_metadata = [
            {'detection_out': {'dtype': 1, 'shape': [1, 1, 200, 7]}},
            {'detection_out': {'dtype': 1, 'shape': [1, 1, 200, 7]}},
            {'age': {'dtype': 1, 'shape': [1, 1, 1, 1]},
             'gender': {'dtype': 1, 'shape': [1, 2, 1, 1]}}]
        expected_inputs_metadata = [
            {'data': {'dtype': 1, 'shape': [1, 3, 300, 300]}},
            {'data': {'dtype': 1, 'shape': [1, 3, 1024, 1024]}},
            {'new_key': {'dtype': 1, 'shape': [1, 3, 62, 62]}}]
        for x in range(len(versions)):
            print("Getting info about model version:".format(
                versions[x]))
            expected_input_metadata = expected_inputs_metadata[x]
            expected_output_metadata = expected_outputs_metadata[x]
            rest_url = 'http://localhost:5560/v1/models/{}/' \
                       'versions/{}/metadata'.format(model_name, versions[x])
            result = requests.get(rest_url)
            print(result.text)
            if not throw_error[x]:
                output_json = result.text
                metadata_pb = get_model_metadata_pb2. \
                    GetModelMetadataResponse()
                response = Parse(output_json, metadata_pb,
                                 ignore_unknown_fields=False)
                input_metadata, output_metadata = model_metadata_response(
                    response=response)

                print(output_metadata)
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
    def test_get_model_status_rest(self, model_version_policy_models,
                                   start_server_model_ver_policy,
                                   model_name, throw_error):

        print("Downloaded model files:", model_version_policy_models)

        versions = [1, 2, 3]
        for x in range(len(versions)):
            rest_url = 'http://localhost:5560/v1/models/{}/' \
                       'versions/{}'.format(model_name, versions[x])
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
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
            else:
                assert 404 == result.status_code

                #   aggregated results check
        if model_name == 'all':
            rest_url = 'http://localhost:5560/v1/models/all'
            response = get_model_status_response_rest(rest_url)
            versions_statuses = response.model_version_status
            assert len(versions_statuses) == 3
            for version_status in versions_statuses:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
