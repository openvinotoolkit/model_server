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
import time
import shutil
from conftest import copy_model, get_model_metadata, model_metadata_response, \
    get_model_metadata_response_rest # noqa


class TestSingleModelInference():

    def test_latest_version(self, download_two_model_versions, get_test_dir,
                            start_server_update_flow_latest,
                            create_channel_for_update_flow_latest):
        resnet_v1, resnet_v2 = download_two_model_versions
        dir = get_test_dir + '/saved_models/' + 'update/'
        resnet_v1_copy_dir = copy_model(resnet_v1, 1, dir)
        time.sleep(8)
        stub = create_channel_for_update_flow_latest

        print("Getting info about resnet model")
        model_name = 'resnet'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1000]}}
        request = get_model_metadata(model_name=model_name)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
        shutil.rmtree(resnet_v1_copy_dir)
        resnet_v2_copy_dir = copy_model(resnet_v2, 2, dir)
        time.sleep(3)

        out_name = 'resnet_v2_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
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
        shutil.rmtree(resnet_v2_copy_dir)

    def test_latest_version_rest(self, download_two_model_versions,
                                 get_test_dir,
                                 start_server_update_flow_latest):
        resnet_v1, resnet_v2 = download_two_model_versions
        dir = get_test_dir + '/saved_models/' + 'update/'
        resnet_v1_copy_dir = copy_model(resnet_v1, 1, dir)
        time.sleep(8)

        print("Getting info about resnet model")
        model_name = 'resnet'
        out_name = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1000]}}
        rest_url = 'http://localhost:5562/v1/models/resnet/metadata'
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
        shutil.rmtree(resnet_v1_copy_dir)
        resnet_v2_copy_dir = copy_model(resnet_v2, 2, dir)
        time.sleep(3)

        out_name = 'resnet_v2_50/predictions/Reshape_1'
        expected_input_metadata = {'input': {'dtype': 1,
                                             'shape': [1, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [1, 1001]}}
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata
        shutil.rmtree(resnet_v2_copy_dir)

    def test_specific_version(self, download_two_model_versions,
                              resnet_2_out_model_downloader, get_test_dir,
                              start_server_update_flow_specific,
                              create_channel_for_update_flow_specific):
        resnet_v1, resnet_v2 = download_two_model_versions
        resnet_2_out = resnet_2_out_model_downloader
        dir = get_test_dir + '/saved_models/' + 'update/'
        stub = create_channel_for_update_flow_specific
        resnet_v1_copy_dir = copy_model(resnet_v1, 1, dir)
        resnet_2_out_copy_dir = copy_model(resnet_2_out, 4, dir)
        time.sleep(8)

        # Available versions: 1, 4

        print("Getting info about resnet model")
        model_name = 'resnet'
        out_name_v1 = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata_v1 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name_v1: {'dtype': 1,
                                                     'shape': [1, 1000]}}
        request = get_model_metadata(model_name=model_name, version=1)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        request_latest = get_model_metadata(model_name=model_name)
        response_latest = stub.GetModelMetadata(request_latest, 10)
        print("response", response_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)

        request_v4 = get_model_metadata(model_name=model_name, version=4)
        response_v4 = stub.GetModelMetadata(request_v4, 10)
        print("response", response_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_latest)

        assert response_v4.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v4 == input_metadata_latest
        assert output_metadata_v4 == output_metadata_latest

        shutil.rmtree(resnet_2_out_copy_dir)
        resnet_v2_copy_dir = copy_model(resnet_v2, 3, dir)
        time.sleep(3)

        # Available versions: 1, 3

        request_latest = get_model_metadata(model_name=model_name)
        response_latest = stub.GetModelMetadata(request_latest, 10)
        print("response", response_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)

        request_v3 = get_model_metadata(model_name=model_name, version=3)
        response_v3 = stub.GetModelMetadata(request_v3, 10)
        print("response", response_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert response_v3.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v3 == input_metadata_latest
        assert output_metadata_v3 == output_metadata_latest

        # Available versions: 1, 3, 4

        resnet_2_out_copy_dir = copy_model(resnet_2_out, 4, dir)
        time.sleep(3)

        request_v1 = get_model_metadata(model_name=model_name, version=1)
        response_v1 = stub.GetModelMetadata(request_v1, 10)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        out_name_v3 = 'resnet_v2_50/predictions/Reshape_1'
        expected_input_metadata_v3 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v3 = {out_name_v3: {'dtype': 1,
                                                     'shape': [1, 1001]}}

        request_v3 = get_model_metadata(model_name=model_name, version=3)
        response_v3 = stub.GetModelMetadata(request_v3, 10)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v4 = {
            'res5c_branch2c1': {'dtype': 1, 'shape': [1, 2048, 7, 7]},
            'res5c_branch2c2': {'dtype': 1, 'shape': [1, 2048, 7, 7]}}
        request_v4 = get_model_metadata(model_name=model_name)
        response_v4 = stub.GetModelMetadata(request_v4, 10)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert model_name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        shutil.rmtree(resnet_v2_copy_dir)
        shutil.rmtree(resnet_v1_copy_dir)
        shutil.rmtree(resnet_2_out_copy_dir)

    def test_specific_version_rest(self, download_two_model_versions,
                                   resnet_2_out_model_downloader,
                                   get_test_dir,
                                   start_server_update_flow_specific):
        resnet_v1, resnet_v2 = download_two_model_versions
        resnet_2_out = resnet_2_out_model_downloader
        dir = get_test_dir + '/saved_models/' + 'update/'
        resnet_v1_copy_dir = copy_model(resnet_v1, 1, dir)
        resnet_2_out_copy_dir = copy_model(resnet_2_out, 4, dir)
        time.sleep(8)

        # Available versions: 1, 4

        print("Getting info about resnet model")
        model_name = 'resnet'
        out_name_v1 = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata_v1 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name_v1: {'dtype': 1,
                                                     'shape': [1, 1000]}}
        rest_url_latest = 'http://localhost:5563/v1/models/resnet/' \
                          'versions/1/metadata'
        response = get_model_metadata_response_rest(rest_url_latest)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        rest_url = 'http://localhost:5563/v1/models/resnet/metadata'
        response_latest = get_model_metadata_response_rest(rest_url)
        print("response", response_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)

        rest_url_v4 = 'http://localhost:5563/v1/models/resnet/' \
                      'versions/1/metadata'
        response_v4 = get_model_metadata_response_rest(rest_url_v4)
        print("response", response_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_latest)

        assert response_v4.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v4 == input_metadata_latest
        assert output_metadata_v4 == output_metadata_latest

        shutil.rmtree(resnet_2_out_copy_dir)
        resnet_v2_copy_dir = copy_model(resnet_v2, 3, dir)
        time.sleep(3)

        # Available versions: 1, 3

        rest_url = 'http://localhost:5563/v1/models/resnet/metadata'
        response_latest = get_model_metadata_response_rest(rest_url)
        print("response", response_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)

        rest_url_v3 = 'http://localhost:5563/v1/models/resnet/' \
                      'versions/3/metadata'
        response_v3 = get_model_metadata_response_rest(rest_url_v3)
        print("response", response_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert response_v3.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v3 == input_metadata_latest
        assert output_metadata_v3 == output_metadata_latest

        # Available versions: 1, 3, 4

        resnet_2_out_copy_dir = copy_model(resnet_2_out, 4, dir)
        time.sleep(3)

        rest_url_v1 = 'http://localhost:5563/v1/models/resnet/' \
                      'versions/1/metadata'
        response_v1 = get_model_metadata_response_rest(rest_url_v1)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        out_name_v3 = 'resnet_v2_50/predictions/Reshape_1'
        expected_input_metadata_v3 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v3 = {out_name_v3: {'dtype': 1,
                                                     'shape': [1, 1001]}}

        rest_url_v3 = 'http://localhost:5563/v1/models/resnet/' \
                      'versions/3/metadata'
        response_v3 = get_model_metadata_response_rest(rest_url_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v4 = {
            'res5c_branch2c1': {'dtype': 1, 'shape': [1, 2048, 7, 7]},
            'res5c_branch2c2': {'dtype': 1, 'shape': [1, 2048, 7, 7]}}
        rest_url = 'http://localhost:5563/v1/models/resnet/metadata'
        response_v4 = get_model_metadata_response_rest(rest_url)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert model_name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        shutil.rmtree(resnet_v2_copy_dir)
        shutil.rmtree(resnet_v1_copy_dir)
        shutil.rmtree(resnet_2_out_copy_dir)

    def test_update_rest_grpc(self, download_two_model_versions,
                              resnet_2_out_model_downloader, get_test_dir,
                              start_server_update_flow_specific,
                              create_channel_for_update_flow_specific):
        resnet_v1, resnet_v2 = download_two_model_versions
        resnet_2_out = resnet_2_out_model_downloader
        dir = get_test_dir + '/saved_models/' + 'update/'
        stub = create_channel_for_update_flow_specific
        resnet_v1_copy_dir = copy_model(resnet_v1, 1, dir)
        resnet_2_out_copy_dir = copy_model(resnet_2_out, 4, dir)
        time.sleep(8)

        # Available versions: 1, 4

        print("Getting info about resnet model")
        model_name = 'resnet'
        out_name_v1 = 'resnet_v1_50/predictions/Reshape_1'
        expected_input_metadata_v1 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name_v1: {'dtype': 1,
                                                     'shape': [1, 1000]}}
        request = get_model_metadata(model_name=model_name, version=1)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        rest_url = 'http://localhost:5563/v1/models/resnet/metadata'
        response_latest = get_model_metadata_response_rest(rest_url)
        print("response", response_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)

        request_v4 = get_model_metadata(model_name=model_name, version=4)
        response_v4 = stub.GetModelMetadata(request_v4, 10)
        print("response", response_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_latest)

        assert response_v4.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v4 == input_metadata_latest
        assert output_metadata_v4 == output_metadata_latest

        shutil.rmtree(resnet_2_out_copy_dir)
        resnet_v2_copy_dir = copy_model(resnet_v2, 3, dir)
        time.sleep(3)

        # Available versions: 1, 3

        request_latest = get_model_metadata(model_name=model_name)
        response_latest = stub.GetModelMetadata(request_latest, 10)
        print("response", response_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)

        rest_url = 'http://localhost:5563/v1/models/resnet/versions/3/metadata'
        response_v3 = get_model_metadata_response_rest(rest_url)
        print("response", response_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert response_v3.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v3 == input_metadata_latest
        assert output_metadata_v3 == output_metadata_latest

        # Available versions: 1, 3, 4

        resnet_2_out_copy_dir = copy_model(resnet_2_out, 4, dir)
        time.sleep(3)
        rest_url = 'http://localhost:5563/v1/models/resnet/versions/1/metadata'
        response_v1 = get_model_metadata_response_rest(rest_url)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        out_name_v3 = 'resnet_v2_50/predictions/Reshape_1'
        expected_input_metadata_v3 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v3 = {out_name_v3: {'dtype': 1,
                                                     'shape': [1, 1001]}}

        request_v3 = get_model_metadata(model_name=model_name, version=3)
        response_v3 = stub.GetModelMetadata(request_v3, 10)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {'input': {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v4 = {
            'res5c_branch2c1': {'dtype': 1, 'shape': [1, 2048, 7, 7]},
            'res5c_branch2c2': {'dtype': 1, 'shape': [1, 2048, 7, 7]}}
        rest_url = 'http://localhost:5563/v1/models/resnet/versions/4/metadata'
        response_v4 = get_model_metadata_response_rest(rest_url)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert model_name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        shutil.rmtree(resnet_v2_copy_dir)
        shutil.rmtree(resnet_v1_copy_dir)
        shutil.rmtree(resnet_2_out_copy_dir)
