#
# Copyright (c) 2021 Intel Corporation
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

from grpc import ChannelCredentials, ssl_channel_credentials
from grpc._channel import Channel
import pytest
from ovmsclient.tfs_compat.grpc.serving_client import GrpcClient
from config import BUILD_INVALID, BUILD_VALID
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

@pytest.mark.parametrize("config, method_call_count", BUILD_VALID)
def test_build_valid(mocker, config, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_config')
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._prepare_certs', return_value=(b'server_certificate', b'client_certificate', b'client_key'))

    client = GrpcClient._build(config)

    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']
    assert type(client.channel) == Channel
    assert type(client.model_service_stub) == ModelServiceStub
    assert type(client.prediction_service_stub) == PredictionServiceStub

@pytest.mark.parametrize("config, expected_exception, expected_message, method_call_count", BUILD_INVALID)
def test_build_invalid(mocker, config, expected_exception, expected_message, method_call_count):
    mock_check_config = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._check_config', side_effect=expected_exception(expected_message))
    mock_prepare_certs = mocker.patch('ovmsclient.tfs_compat.grpc.serving_client._prepare_certs')

    with pytest.raises(expected_exception) as e_info:
        GrpcClient._build(config)
        assert str(e_info.value) == expected_message

    assert mock_check_config.call_count == method_call_count['check_config']
    assert mock_prepare_certs.call_count == method_call_count['prepare_certs']
