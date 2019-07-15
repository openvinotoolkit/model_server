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

import grpc
from conftest import get_fake_model_status_request, MODEL_SERVICE


def test_get_model_metadata_wrong_model_version(
        get_grpc_service_for_model_status):
    wrong_requested_version = 999
    request = get_fake_model_status_request(model_name='test',
                                            version=wrong_requested_version)
    grpc_server = get_grpc_service_for_model_status
    rpc = grpc_server.invoke_unary_unary(
        MODEL_SERVICE.methods_by_name['GetModelStatus'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.NOT_FOUND == code


def test_get_model_metadata_wrong_model(get_grpc_service_for_model_status):
    wrong_model_name = "wrong"
    request = get_fake_model_status_request(model_name=wrong_model_name,
                                            version=1)
    grpc_server = get_grpc_service_for_model_status
    rpc = grpc_server.invoke_unary_unary(
        MODEL_SERVICE.methods_by_name['GetModelStatus'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.NOT_FOUND == code


def test_get_model_metadata_correct_response(
        get_grpc_service_for_model_status):
    request = get_fake_model_status_request(model_name='test',
                                            version=1)
    grpc_server = get_grpc_service_for_model_status
    rpc = grpc_server.invoke_unary_unary(
        MODEL_SERVICE.methods_by_name['GetModelStatus'],
        (),
        request, None)
    rpc.initial_metadata()
    response, trailing_metadata, code, details = rpc.termination()
    assert grpc.StatusCode.OK == code
