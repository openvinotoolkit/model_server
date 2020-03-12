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
import shutil
import sys
import time

from constants import PREDICTION_SERVICE, MODEL_SERVICE
from utils.grpc import get_model_metadata, model_metadata_response, \
    get_model_status
from utils.model_management import copy_model
from utils.rest import get_model_metadata_response_rest, \
    get_model_status_response_rest

sys.path.append(".")
from ie_serving.models.models_utils import ModelVersionState, ErrorCode, \
    _ERROR_MESSAGE  # noqa


class TestSingleModelInference():

    def test_specific_version(self, resnet_multiple_batch_sizes, get_test_dir,
                              start_server_update_flow_specific,
                              create_grpc_channel):
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        dir = get_test_dir + '/saved_models/' + 'update/'
        # ensure model dir is empty at the beginning
        shutil.rmtree(dir, ignore_errors=True)
        stub = create_grpc_channel('localhost:9008', PREDICTION_SERVICE)
        status_stub = create_grpc_channel('localhost:9008', MODEL_SERVICE)

        resnet_copy_dir = copy_model(resnet, 1, dir)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, dir)

        # This could be replaced with status polling
        time.sleep(8)

        # Available versions: 1, 4

        print("Getting info about resnet model")
        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata_v1 = {in_name: {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name: {'dtype': 1,
                                                  'shape': [1, 1001]}}
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

        # Model status check
        model_name = 'resnet'
        request = get_model_status(model_name=model_name)
        status_response = status_stub.GetModelStatus(request, 10)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 2
        for version_status in versions_statuses:
            assert version_status.version in [1, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == _ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_bs4_copy_dir)
        resnet_bs8_copy_dir = copy_model(resnet_bs8, 3, dir)
        time.sleep(10)

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

        # Model status check
        model_name = 'resnet'
        request = get_model_status(model_name=model_name)
        status_response = status_stub.GetModelStatus(request, 10)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            if version_status.version == 4:
                assert version_status.state == ModelVersionState.END
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 1 or version_status.version == 3:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        # Available versions: 1, 3, 4

        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, dir)
        time.sleep(10)

        request_v1 = get_model_metadata(model_name=model_name, version=1)
        response_v1 = stub.GetModelMetadata(request_v1, 10)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert model_name == response_v1.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        expected_input_metadata_v3 = {in_name: {'dtype': 1,
                                                'shape': [8, 3, 224, 224]}}
        expected_output_metadata_v3 = {out_name: {'dtype': 1,
                                                  'shape': [8, 1001]}}

        request_v3 = get_model_metadata(model_name=model_name, version=3)
        response_v3 = stub.GetModelMetadata(request_v3, 10)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert model_name == response_v3.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {in_name: {'dtype': 1,
                                                'shape': [4, 3, 224, 224]}}
        expected_output_metadata_v4 = {out_name: {'dtype': 1,
                                                  'shape': [4, 1001]}}
        request_v4 = get_model_metadata(model_name=model_name)
        response_v4 = stub.GetModelMetadata(request_v4, 10)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert model_name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        # Model status check
        model_name = 'resnet'
        request = get_model_status(model_name=model_name)
        status_response = status_stub.GetModelStatus(request, 10)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == _ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_copy_dir)
        shutil.rmtree(resnet_bs4_copy_dir)
        shutil.rmtree(resnet_bs8_copy_dir)
        time.sleep(10)

    def test_latest_version(self, resnet_multiple_batch_sizes, get_test_dir,
                            start_server_update_flow_latest,
                            create_grpc_channel):

        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        dir = get_test_dir + '/saved_models/' + 'update/'
        # ensure model dir is empty at the beginning
        shutil.rmtree(dir, ignore_errors=True)
        resnet_v1_copy_dir = copy_model(resnet, 1, dir)
        time.sleep(8)
        stub = create_grpc_channel('localhost:9007', PREDICTION_SERVICE)
        status_stub = create_grpc_channel('localhost:9007', MODEL_SERVICE)

        print("Getting info about resnet model")
        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata_v1 = {in_name: {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name: {'dtype': 1,
                                                  'shape': [1, 1001]}}
        request = get_model_metadata(model_name=model_name)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        # Model status check before update
        model_name = 'resnet'
        request = get_model_status(model_name=model_name)
        status_response = status_stub.GetModelStatus(request, 10)
        versions_statuses = status_response.model_version_status
        version_status = versions_statuses[0]
        assert len(versions_statuses) == 1
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        resnet_v2_copy_dir = copy_model(resnet_bs4, 2, dir)
        time.sleep(10)

        expected_input_metadata_v2 = {in_name: {'dtype': 1,
                                                'shape': [4, 3, 224, 224]}}
        expected_output_metadata_v2 = {out_name: {'dtype': 1,
                                                  'shape': [4, 1001]}}
        request = get_model_metadata(model_name=model_name)
        response = stub.GetModelMetadata(request, 10)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata_v2 == input_metadata
        assert expected_output_metadata_v2 == output_metadata

        # Model status check after update
        model_name = 'resnet'
        request = get_model_status(model_name=model_name)
        status_response = status_stub.GetModelStatus(request, 10)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 2
        for version_status in versions_statuses:
            assert version_status.version in [1, 2]
            if version_status.version == 1:
                assert version_status.state == ModelVersionState.END
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 2:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###
        shutil.rmtree(resnet_v1_copy_dir)
        shutil.rmtree(resnet_v2_copy_dir)
        time.sleep(10)

    def test_specific_version_rest(self, resnet_multiple_batch_sizes,
                                   get_test_dir,
                                   start_server_update_flow_specific):
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        dir = get_test_dir + '/saved_models/' + 'update/'
        # ensure model dir is empty at the beginning
        shutil.rmtree(dir, ignore_errors=True)
        resnet_copy_dir = copy_model(resnet, 1, dir)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, dir)
        time.sleep(8)

        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'

        # Available versions: 1, 4

        print("Getting info about resnet model")
        model_name = 'resnet'

        expected_input_metadata_v1 = {in_name: {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name: {'dtype': 1,
                                                  'shape': [1, 1001]}}

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
                      'versions/4/metadata'
        response_v4 = get_model_metadata_response_rest(rest_url_v4)
        print("response", response_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_latest)

        assert response_v4.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v4 == input_metadata_latest
        assert output_metadata_v4 == output_metadata_latest

        # Model status check
        rest_status_url = 'http://localhost:5563/v1/models/resnet'
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 2
        for version_status in versions_statuses:
            assert version_status.version in [1, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == _ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_bs4_copy_dir)
        resnet_bs8_copy_dir = copy_model(resnet_bs8, 3, dir)
        time.sleep(10)

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

        # Model status check
        rest_status_url = 'http://localhost:5563/v1/models/resnet'
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            if version_status.version == 4:
                assert version_status.state == ModelVersionState.END
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 1 or version_status.version == 3:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        # Available versions: 1, 3, 4

        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, dir)
        time.sleep(10)

        rest_url_v1 = 'http://localhost:5563/v1/models/resnet/' \
                      'versions/1/metadata'
        response_v1 = get_model_metadata_response_rest(rest_url_v1)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert model_name == response_v1.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        expected_input_metadata_v3 = {in_name: {'dtype': 1,
                                                'shape': [8, 3, 224, 224]}}
        expected_output_metadata_v3 = {out_name: {'dtype': 1,
                                                  'shape': [8, 1001]}}

        rest_url_v3 = 'http://localhost:5563/v1/models/resnet/' \
                      'versions/3/metadata'
        response_v3 = get_model_metadata_response_rest(rest_url_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert model_name == response_v3.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {in_name: {'dtype': 1,
                                                'shape': [4, 3, 224, 224]}}
        expected_output_metadata_v4 = {out_name: {'dtype': 1,
                                                  'shape': [4, 1001]}}
        response_v4 = get_model_metadata_response_rest(rest_url)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert model_name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        # Model status check
        rest_status_url = 'http://localhost:5563/v1/models/resnet'
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == _ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_copy_dir)
        shutil.rmtree(resnet_bs4_copy_dir)
        shutil.rmtree(resnet_bs8_copy_dir)
        time.sleep(10)

    def test_latest_version_rest(self, resnet_multiple_batch_sizes,
                                 get_test_dir,
                                 start_server_update_flow_latest):
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        dir = get_test_dir + '/saved_models/' + 'update/'
        # ensure model dir is empty at the beginning
        shutil.rmtree(dir, ignore_errors=True)
        resnet_copy_dir = copy_model(resnet, 1, dir)
        time.sleep(8)

        print("Getting info about resnet model")
        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata_v1 = {in_name: {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name: {'dtype': 1,
                                                  'shape': [1, 1001]}}

        rest_url = 'http://localhost:5562/v1/models/resnet/metadata'
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        # Model status check before update
        rest_status_url = 'http://localhost:5562/v1/models/resnet'
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        version_status = versions_statuses[0]
        assert len(versions_statuses) == 1
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == _ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_copy_dir)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 2, dir)
        time.sleep(10)

        expected_input_metadata = {in_name: {'dtype': 1,
                                             'shape': [4, 3, 224, 224]}}
        expected_output_metadata = {out_name: {'dtype': 1,
                                               'shape': [4, 1001]}}
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        print(output_metadata)
        assert model_name == response.model_spec.name
        assert expected_input_metadata == input_metadata
        assert expected_output_metadata == output_metadata

        # Model status check after update
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 2
        for version_status in versions_statuses:
            assert version_status.version in [1, 2]
            if version_status.version == 1:
                assert version_status.state == ModelVersionState.END
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 2:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == _ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_bs4_copy_dir)
        time.sleep(10)

    def test_update_rest_grpc(self, resnet_multiple_batch_sizes, get_test_dir,
                              start_server_update_flow_specific,
                              create_grpc_channel):
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        dir = get_test_dir + '/saved_models/' + 'update/'
        # ensure model dir is empty at the beginning
        shutil.rmtree(dir, ignore_errors=True)
        stub = create_grpc_channel('localhost:9008', PREDICTION_SERVICE)
        resnet_copy_dir = copy_model(resnet, 1, dir)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, dir)
        time.sleep(8)

        # Available versions: 1, 4

        print("Getting info about resnet model")
        model_name = 'resnet'
        in_name = 'map/TensorArrayStack/TensorArrayGatherV3'
        out_name = 'softmax_tensor'
        expected_input_metadata_v1 = {in_name: {'dtype': 1,
                                                'shape': [1, 3, 224, 224]}}
        expected_output_metadata_v1 = {out_name: {'dtype': 1,
                                                  'shape': [1, 1001]}}
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

        shutil.rmtree(resnet_bs4_copy_dir)
        resnet_bs8_copy_dir = copy_model(resnet_bs8, 3, dir)
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

        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, dir)
        time.sleep(3)
        rest_url = 'http://localhost:5563/v1/models/resnet/versions/1/metadata'
        response_v1 = get_model_metadata_response_rest(rest_url)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        expected_input_metadata_v3 = {in_name: {'dtype': 1,
                                                'shape': [8, 3, 224, 224]}}
        expected_output_metadata_v3 = {out_name: {'dtype': 1,
                                                  'shape': [8, 1001]}}

        request_v3 = get_model_metadata(model_name=model_name, version=3)
        response_v3 = stub.GetModelMetadata(request_v3, 10)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert model_name == response.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {in_name: {'dtype': 1,
                                                'shape': [4, 3, 224, 224]}}
        expected_output_metadata_v4 = {out_name: {'dtype': 1,
                                                  'shape': [4, 1001]}}
        rest_url = 'http://localhost:5563/v1/models/resnet/versions/4/metadata'
        response_v4 = get_model_metadata_response_rest(rest_url)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert model_name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        shutil.rmtree(resnet_copy_dir)
        shutil.rmtree(resnet_bs4_copy_dir)
        shutil.rmtree(resnet_bs8_copy_dir)
