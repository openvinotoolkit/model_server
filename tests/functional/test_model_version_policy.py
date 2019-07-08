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
import requests
import pytest
from google.protobuf.json_format import Parse
from grpc.framework.interfaces.face.face import AbortionError
sys.path.append(".")
from conftest import get_model_metadata, model_metadata_response  # noqa
from ie_serving.tensorflow_serving_api import get_model_metadata_pb2  # noqa


class TestModelVerPolicy():

    @pytest.mark.parametrize("model_name, throw_error", [
        ('all', [False, False, False]),
        ('specific', [False, True, False]),
        ('latest', [True, False, False]),
    ])
    def test_get_model_metadata(self, model_version_policy_models,
                                start_server_model_ver_policy,
                                create_channel_for_model_ver_pol_server,
                                model_name, throw_error):
        """
        <b>Description</b>
        Execute GetModelMetadata request using gRPC interface
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

        # Connect to grpc service
        stub = create_channel_for_model_ver_pol_server

        print("Getting info about resnet model")
        versions = [1, 2, 3]
        expected_outputs_metadata = [
            {'resnet_v1_50/predictions/Reshape_1': {'dtype': 1,
                                                    'shape': [1, 1000]}},
            {'resnet_v2_50/predictions/Reshape_1': {'dtype': 1,
                                                    'shape': [1, 1001]}},
            {'mask': {'dtype': 1, 'shape': [1, 2048, 7, 7]},
             'output': {'dtype': 1, 'shape': [1, 2048, 7, 7]}}]
        expected_inputs_metadata = [
            {'input': {'dtype': 1, 'shape': [1, 3, 224, 224]}},
            {'input': {'dtype': 1, 'shape': [1, 3, 224, 224]}},
            {'new_key': {'dtype': 1, 'shape': [1, 3, 224, 224]}}]
        for x in range(len(versions)):
            print("Getting info about resnet model version:".format(
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
                with pytest.raises(AbortionError):
                    response = stub.GetModelMetadata(request, 10)

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
        Execute GetModelMetadata request using gRPC interface
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

        print("Getting info about resnet model")
        versions = [1, 2, 3]
        expected_outputs_metadata = [
            {'resnet_v1_50/predictions/Reshape_1': {'dtype': 1,
                                                    'shape': [1, 1000]}},
            {'resnet_v2_50/predictions/Reshape_1': {'dtype': 1,
                                                    'shape': [1, 1001]}},
            {'mask': {'dtype': 1, 'shape': [1, 2048, 7, 7]},
             'output': {'dtype': 1, 'shape': [1, 2048, 7, 7]}}]
        expected_inputs_metadata = [
            {'input': {'dtype': 1, 'shape': [1, 3, 224, 224]}},
            {'input': {'dtype': 1, 'shape': [1, 3, 224, 224]}},
            {'new_key': {'dtype': 1, 'shape': [1, 3, 224, 224]}}]
        for x in range(len(versions)):
            print("Getting info about resnet model version:".format(
                versions[x]))
            expected_input_metadata = expected_inputs_metadata[x]
            expected_output_metadata = expected_outputs_metadata[x]
            rest_url = 'http://localhost:5560/v1/models/{}/' \
                       'versions/{}/metadata'.format(model_name, versions[x])
            result = requests.get(rest_url)
            print(result.text)
            if not throw_error[x]:
                output_json = result.text
                metadata_pb = get_model_metadata_pb2.\
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
