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
import os
import shutil
import time
import pytest

import config
from constants import MODEL_SERVICE, NOT_TO_BE_REPORTED_IF_SKIPPED
from model.models_information import Resnet, ResnetBS8, ResnetBS4
from utils.grpc import create_channel, get_model_metadata_request, get_model_metadata, model_metadata_response, \
    get_model_status
import logging
from utils.model_management import copy_model
from utils.rest import get_metadata_url, get_status_url, get_model_metadata_response_rest, \
    get_model_status_response_rest
from utils.parametrization import get_tests_suffix


from utils.models_utils import ModelVersionState, ErrorCode, \
    ERROR_MESSAGE  # noqa

logger = logging.getLogger(__name__)


@pytest.mark.skipif(config.skip_nginx_test, reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
class TestSingleModelInference:

    @staticmethod
    def get_update_directory():
        return os.path.join(config.path_to_mount, "update-{}".format(get_tests_suffix()))

    @pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
    def test_specific_version(self, resnet_multiple_batch_sizes, start_server_update_flow_specific):
        _, ports = start_server_update_flow_specific
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        directory = self.get_update_directory()

        # ensure model directory is empty at the beginning
        shutil.rmtree(directory, ignore_errors=True)
        stub = create_channel(port=ports["grpc_port"])
        status_stub = create_channel(port=ports["grpc_port"], service=MODEL_SERVICE)

        resnet_copy_dir = copy_model(resnet, 1, directory)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, directory)

        # This could be replaced with status polling
        time.sleep(8)

        # Available versions: 1, 4

        logger.info("Getting info about {} model".format(Resnet.name))
        expected_input_metadata_v1 = {Resnet.input_name: {'dtype': 1, 'shape': list(Resnet.input_shape)}}
        expected_output_metadata_v1 = {Resnet.output_name: {'dtype': 1, 'shape': list(Resnet.output_shape)}}
        request = get_model_metadata_request(model_name=Resnet.name, version=1)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        request_latest = get_model_metadata_request(model_name=Resnet.name)
        response_latest = get_model_metadata(stub, request_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_latest))
        logger.info("Output metadata: {}".format(output_metadata_latest))

        request_v4 = get_model_metadata_request(model_name=Resnet.name, version=4)
        response_v4 = get_model_metadata(stub, request_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_v4))
        logger.info("Output metadata: {}".format(output_metadata_v4))

        assert response_v4.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v4 == input_metadata_latest
        assert output_metadata_v4 == output_metadata_latest

        # Model status check
        request = get_model_status(model_name=Resnet.name)
        status_response = status_stub.GetModelStatus(request, 60)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 2
        for version_status in versions_statuses:
            assert version_status.version in [1, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_bs4_copy_dir)
        resnet_bs8_copy_dir = copy_model(resnet_bs8, 3, directory)
        time.sleep(10)

        # Available versions: 1, 3

        request_latest = get_model_metadata_request(model_name=Resnet.name)
        response_latest = get_model_metadata(stub, request_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_latest))
        logger.info("Output metadata: {}".format(output_metadata_latest))

        request_v3 = get_model_metadata_request(model_name=Resnet.name, version=3)
        response_v3 = get_model_metadata(stub, request_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)
        logger.info("Input metadata: {}".format(input_metadata_v3))
        logger.info("Output metadata: {}".format(output_metadata_v3))

        assert response_v3.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v3 == input_metadata_latest
        assert output_metadata_v3 == output_metadata_latest

        # Model status check
        request = get_model_status(model_name=Resnet.name)
        status_response = status_stub.GetModelStatus(request, 60)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            if version_status.version == 4:
                assert version_status.state == ModelVersionState.END
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 1 or version_status.version == 3:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        # Available versions: 1, 3, 4

        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, directory)
        time.sleep(10)

        request_v1 = get_model_metadata_request(model_name=Resnet.name, version=1)
        response_v1 = get_model_metadata(stub, request_v1)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert Resnet.name == response_v1.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        expected_input_metadata_v3 = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS8.input_shape)}}
        expected_output_metadata_v3 = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS8.output_shape)}}

        request_v3 = get_model_metadata_request(model_name=Resnet.name, version=3)
        response_v3 = get_model_metadata(stub, request_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert Resnet.name == response_v3.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS4.input_shape)}}
        expected_output_metadata_v4 = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS4.output_shape)}}
        request_v4 = get_model_metadata_request(model_name=Resnet.name)
        response_v4 = get_model_metadata(stub, request_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert Resnet.name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        # Model status check
        request = get_model_status(model_name=Resnet.name)
        status_response = status_stub.GetModelStatus(request, 60)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_copy_dir)
        shutil.rmtree(resnet_bs4_copy_dir)
        shutil.rmtree(resnet_bs8_copy_dir)
        time.sleep(10)

    @pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
    def test_latest_version(self, resnet_multiple_batch_sizes,
                            start_server_update_flow_latest):

        _, ports = start_server_update_flow_latest
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        directory = self.get_update_directory()
        # ensure model directory is empty at the beginning
        shutil.rmtree(directory, ignore_errors=True)
        resnet_v1_copy_dir = copy_model(resnet, 1, directory)
        time.sleep(8)
        stub = create_channel(port=ports["grpc_port"])
        status_stub = create_channel(port=ports["grpc_port"], service=MODEL_SERVICE)

        logger.info("Getting info about {} model".format(Resnet.name))
        expected_input_metadata_v1 = {Resnet.input_name: {'dtype': 1, 'shape': list(Resnet.input_shape)}}
        expected_output_metadata_v1 = {Resnet.output_name: {'dtype': 1, 'shape': list(Resnet.output_shape)}}
        request = get_model_metadata_request(model_name=Resnet.name)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        # Model status check before update
        request = get_model_status(model_name=Resnet.name)
        status_response = status_stub.GetModelStatus(request, 60)
        versions_statuses = status_response.model_version_status
        version_status = versions_statuses[0]
        assert len(versions_statuses) == 1
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        resnet_v2_copy_dir = copy_model(resnet_bs4, 2, directory)
        time.sleep(10)

        expected_input_metadata_v2 = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS4.input_shape)}}
        expected_output_metadata_v2 = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS4.output_shape)}}
        request = get_model_metadata_request(model_name=Resnet.name)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v2 == input_metadata
        assert expected_output_metadata_v2 == output_metadata

        # Model status check after update
        request = get_model_status(model_name=Resnet.name)
        status_response = status_stub.GetModelStatus(request, 60)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 2
        for version_status in versions_statuses:
            assert version_status.version in [1, 2]
            if version_status.version == 1:
                assert version_status.state == ModelVersionState.END
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 2:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###
        shutil.rmtree(resnet_v1_copy_dir)
        shutil.rmtree(resnet_v2_copy_dir)
        time.sleep(10)

    @pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
    def test_specific_version_rest(self, resnet_multiple_batch_sizes, start_server_update_flow_specific):
        _, ports = start_server_update_flow_specific
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        directory = self.get_update_directory()

        # ensure model directory is empty at the beginning
        shutil.rmtree(directory, ignore_errors=True)
        resnet_copy_dir = copy_model(resnet, 1, directory)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, directory)
        time.sleep(8)

        # Available versions: 1, 4

        logger.info("Getting info about {} model".format(Resnet.name))

        expected_input_metadata_v1 = {Resnet.input_name: {'dtype': 1, 'shape': list(Resnet.input_shape)}}
        expected_output_metadata_v1 = {Resnet.output_name: {'dtype': 1, 'shape': list(Resnet.output_shape)}}

        rest_url_latest = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="1")
        response = get_model_metadata_response_rest(rest_url_latest)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"])
        response_latest = get_model_metadata_response_rest(rest_url)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_latest))
        logger.info("Output metadata: {}".format(output_metadata_latest))

        rest_url_v4 = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="4")
        response_v4 = get_model_metadata_response_rest(rest_url_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_v4))
        logger.info("Output metadata: {}".format(output_metadata_v4))

        assert response_v4.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v4 == input_metadata_latest
        assert output_metadata_v4 == output_metadata_latest

        # Model status check
        rest_status_url = get_status_url(model=Resnet.name, port=ports["rest_port"])
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 2
        for version_status in versions_statuses:
            assert version_status.version in [1, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_bs4_copy_dir)
        resnet_bs8_copy_dir = copy_model(resnet_bs8, 3, directory)
        time.sleep(10)

        # Available versions: 1, 3

        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"])
        response_latest = get_model_metadata_response_rest(rest_url)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_latest))
        logger.info("Output metadata: {}".format(output_metadata_latest))

        rest_url_v3 = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="3")
        response_v3 = get_model_metadata_response_rest(rest_url_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)
        logger.info("Input metadata: {}".format(input_metadata_v3))
        logger.info("Output metadata: {}".format(output_metadata_v3))

        assert response_v3.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v3 == input_metadata_latest
        assert output_metadata_v3 == output_metadata_latest

        # Model status check
        rest_status_url = get_status_url(model=Resnet.name, port=ports["rest_port"])
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            if version_status.version == 4:
                assert version_status.state == ModelVersionState.END
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 1 or version_status.version == 3:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        # Available versions: 1, 3, 4

        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, directory)
        time.sleep(10)

        rest_url_v1 = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="1")
        response_v1 = get_model_metadata_response_rest(rest_url_v1)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert Resnet.name == response_v1.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        expected_input_metadata_v3 = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS8.input_shape)}}
        expected_output_metadata_v3 = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS8.output_shape)}}

        rest_url_v3 = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="3")
        response_v3 = get_model_metadata_response_rest(rest_url_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert Resnet.name == response_v3.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS4.input_shape)}}
        expected_output_metadata_v4 = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS4.output_shape)}}
        response_v4 = get_model_metadata_response_rest(rest_url)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert Resnet.name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        # Model status check
        rest_status_url = get_status_url(model=Resnet.name, port=ports["rest_port"])
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        assert len(versions_statuses) == 3
        for version_status in versions_statuses:
            assert version_status.version in [1, 3, 4]
            assert version_status.state == ModelVersionState.AVAILABLE
            assert version_status.status.error_code == ErrorCode.OK
            assert version_status.status.error_message == ERROR_MESSAGE[
                ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_copy_dir)
        shutil.rmtree(resnet_bs4_copy_dir)
        shutil.rmtree(resnet_bs8_copy_dir)
        time.sleep(10)

    @pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
    def test_latest_version_rest(self, resnet_multiple_batch_sizes, start_server_update_flow_latest):
        _, ports = start_server_update_flow_latest
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        directory = self.get_update_directory()

        # ensure model directory is empty at the beginning
        shutil.rmtree(directory, ignore_errors=True)
        resnet_copy_dir = copy_model(resnet, 1, directory)
        time.sleep(8)

        logger.info("Getting info about {} model".format(Resnet.name))
        expected_input_metadata_v1 = {Resnet.input_name: {'dtype': 1, 'shape': list(Resnet.input_shape)}}
        expected_output_metadata_v1 = {Resnet.output_name: {'dtype': 1, 'shape': list(Resnet.output_shape)}}

        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"])
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))
        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        # Model status check before update
        rest_status_url = get_status_url(model=Resnet.name, port=ports["rest_port"])
        status_response = get_model_status_response_rest(rest_status_url)
        versions_statuses = status_response.model_version_status
        version_status = versions_statuses[0]
        assert len(versions_statuses) == 1
        assert version_status.version == 1
        assert version_status.state == ModelVersionState.AVAILABLE
        assert version_status.status.error_code == ErrorCode.OK
        assert version_status.status.error_message == ERROR_MESSAGE[
            ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_copy_dir)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 2, directory)
        time.sleep(10)

        expected_input_metadata = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS4.input_shape)}}
        expected_output_metadata = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS4.output_shape)}}
        response = get_model_metadata_response_rest(rest_url)
        input_metadata, output_metadata = model_metadata_response(
            response=response)
        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert Resnet.name == response.model_spec.name
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
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.END][ErrorCode.OK]
            elif version_status.version == 2:
                assert version_status.state == ModelVersionState.AVAILABLE
                assert version_status.status.error_code == ErrorCode.OK
                assert version_status.status.error_message == ERROR_MESSAGE[
                    ModelVersionState.AVAILABLE][ErrorCode.OK]
        ###

        shutil.rmtree(resnet_bs4_copy_dir)
        time.sleep(10)

    @pytest.mark.skip(reason=NOT_TO_BE_REPORTED_IF_SKIPPED)
    def test_update_rest_grpc(self, resnet_multiple_batch_sizes, start_server_update_flow_specific):
        _, ports = start_server_update_flow_specific
        resnet, resnet_bs4, resnet_bs8 = resnet_multiple_batch_sizes
        directory = self.get_update_directory()

        # ensure model directory is empty at the beginning
        shutil.rmtree(directory, ignore_errors=True)
        stub = create_channel(port=ports["grpc_port"])
        resnet_copy_dir = copy_model(resnet, 1, directory)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, directory)
        time.sleep(8)

        # Available versions: 1, 4

        logger.info("Getting info about {} model".format(Resnet.name))
        expected_input_metadata_v1 = {Resnet.input_name: {'dtype': 1, 'shape': list(Resnet.input_shape)}}
        expected_output_metadata_v1 = {Resnet.output_name: {'dtype': 1, 'shape': list(Resnet.output_shape)}}
        request = get_model_metadata_request(model_name=Resnet.name, version=1)
        response = get_model_metadata(stub, request)
        input_metadata, output_metadata = model_metadata_response(
            response=response)

        logger.info("Input metadata: {}".format(input_metadata))
        logger.info("Output metadata: {}".format(output_metadata))

        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata
        assert expected_output_metadata_v1 == output_metadata

        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"])
        response_latest = get_model_metadata_response_rest(rest_url)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)

        logger.info("Input metadata: {}".format(input_metadata_latest))
        logger.info("Output metadata: {}".format(output_metadata_latest))

        request_v4 = get_model_metadata_request(model_name=Resnet.name, version=4)
        response_v4 = get_model_metadata(stub, request_v4)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_v4))
        logger.info("Output metadata: {}".format(output_metadata_v4))

        assert response_v4.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v4 == input_metadata_latest
        assert output_metadata_v4 == output_metadata_latest

        shutil.rmtree(resnet_bs4_copy_dir)
        resnet_bs8_copy_dir = copy_model(resnet_bs8, 3, directory)
        time.sleep(3)

        # Available versions: 1, 3

        request_latest = get_model_metadata_request(model_name=Resnet.name)
        response_latest = get_model_metadata(stub, request_latest)
        input_metadata_latest, output_metadata_latest = \
            model_metadata_response(response=response_latest)
        logger.info("Input metadata: {}".format(input_metadata_latest))
        logger.info("Output metadata: {}".format(output_metadata_latest))

        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="3")
        response_v3 = get_model_metadata_response_rest(rest_url)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)
        logger.info("Input metadata: {}".format(input_metadata_v3))
        logger.info("Output metadata: {}".format(output_metadata_v3))

        assert response_v3.model_spec.name == response_latest.model_spec.name
        assert input_metadata_v3 == input_metadata_latest
        assert output_metadata_v3 == output_metadata_latest

        # Available versions: 1, 3, 4

        time.sleep(3)
        resnet_bs4_copy_dir = copy_model(resnet_bs4, 4, directory)
        time.sleep(3)
        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="1")
        response_v1 = get_model_metadata_response_rest(rest_url)
        input_metadata_v1, output_metadata_v1 = model_metadata_response(
            response=response_v1)

        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v1 == input_metadata_v1
        assert expected_output_metadata_v1 == output_metadata_v1

        expected_input_metadata_v3 = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS8.input_shape)}}
        expected_output_metadata_v3 = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS8.output_shape)}}

        request_v3 = get_model_metadata_request(model_name=Resnet.name, version=3)
        response_v3 = get_model_metadata(stub, request_v3)
        input_metadata_v3, output_metadata_v3 = model_metadata_response(
            response=response_v3)

        assert Resnet.name == response.model_spec.name
        assert expected_input_metadata_v3 == input_metadata_v3
        assert expected_output_metadata_v3 == output_metadata_v3

        expected_input_metadata_v4 = {Resnet.input_name: {'dtype': 1, 'shape': list(ResnetBS4.input_shape)}}
        expected_output_metadata_v4 = {Resnet.output_name: {'dtype': 1, 'shape': list(ResnetBS4.output_shape)}}
        rest_url = get_metadata_url(model=Resnet.name, port=ports["rest_port"], version="4")
        response_v4 = get_model_metadata_response_rest(rest_url)
        input_metadata_v4, output_metadata_v4 = model_metadata_response(
            response=response_v4)

        assert Resnet.name == response_v4.model_spec.name
        assert expected_input_metadata_v4 == input_metadata_v4
        assert expected_output_metadata_v4 == output_metadata_v4

        shutil.rmtree(resnet_copy_dir)
        shutil.rmtree(resnet_bs4_copy_dir)
        shutil.rmtree(resnet_bs8_copy_dir)
