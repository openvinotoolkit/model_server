#
# Copyright (c) 2020 Intel Corporation
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

import pytest
import grpc
import numpy as np
from src.api.ovms_connector import OvmsConnector, RequestProcessingError, ModelNotFoundError, OvmsUnavailableError
from tensorflow import make_tensor_proto
from tests.unit.conftest import FakeGrpcStub

input_image = np.zeros((1, 3, 224, 224))
inference_input = {"input": input_image}


@pytest.mark.parametrize("error_type, error_message",
                         [(TypeError, "Unsupported data type"), (ValueError, "Invalid arguments")])
def test_ovms_connector_prepare_input_fail(mocker, error_type, error_message):
    make_tensor_proto_mock = mocker.patch(
        "src.api.ovms_connector.make_tensor_proto")
    make_tensor_proto_mock.side_effect = error_type
    ovms_connector = OvmsConnector(
        "4000", {"model_name": "test", "model_version": 1, "input_name": "input"})
    with pytest.raises(error_type, match=error_message):
        ovms_connector.send(inference_input)


@pytest.mark.parametrize("error_type, error_message",
                         [(ValueError, "Invalid inference input")])
def test_ovms_connector_prepare_input_none(mocker, error_type, error_message):
    make_tensor_proto_mock = mocker.patch(
        "src.api.ovms_connector.make_tensor_proto")
    make_tensor_proto_mock.side_effect = error_type
    ovms_connector = OvmsConnector(
        "4000", {"model_name": "test", "model_version": 1, "input_name": "input"})
    inference_input = None
    with pytest.raises(error_type, match=error_message):
        ovms_connector.send(inference_input)


@pytest.mark.parametrize("grpc_error_code, ams_error_type, ams_error_message", [
    (grpc.StatusCode.INVALID_ARGUMENT, RequestProcessingError,
     "Error during inference request*"),
    (grpc.StatusCode.NOT_FOUND, ModelNotFoundError, "Requested model not found"),
    (grpc.StatusCode.UNAVAILABLE, OvmsUnavailableError, "Unable to connect to OVMS"),
    (-1000, Exception, "GRPC error")
])
def test_ovms_connector_send_request_fail(mocker, grpc_error_code, ams_error_type, ams_error_message):
    inference_input_tensor = make_tensor_proto(input_image)
    make_tensor_proto_mock = mocker.patch(
        "src.api.ovms_connector.make_tensor_proto")
    make_tensor_proto_mock.return_value = inference_input_tensor

    ovms_connector = OvmsConnector(
        "4000", {"model_name": "test", "model_version": 1, "input_name": "input"})
    ovms_connector.stub = FakeGrpcStub(grpc_error_code)
    with pytest.raises(ams_error_type, match=ams_error_message):
        ovms_connector.send(inference_input)


def test_ovms_connector_prepare_output_fail(mocker):
    inference_input_tensor = make_tensor_proto(input_image)
    make_tensor_proto_mock = mocker.patch(
        "src.api.ovms_connector.make_tensor_proto")
    make_tensor_proto_mock.return_value = inference_input_tensor

    ovms_connector = OvmsConnector(
        "4000", {"model_name": "test", "model_version": 1, "input_name": "input"})
    ovms_connector.stub = FakeGrpcStub(None)

    make_ndarray_mock = mocker.patch("src.api.ovms_connector.make_ndarray")
    make_ndarray_mock.side_effect = TypeError
    with pytest.raises(TypeError, match="Output datatype error"):
        ovms_connector.send(inference_input)


def test_ovms_connector_success(mocker):
    expected_result = {"output": np.zeros((1, 1000))}

    inference_input_tensor = make_tensor_proto(input_image)
    make_tensor_proto_mock = mocker.patch(
        "src.api.ovms_connector.make_tensor_proto")
    make_tensor_proto_mock.return_value = inference_input_tensor

    ovms_connector = OvmsConnector(
        "4000", {"model_name": "test", "model_version": 1, "input_name": "input"})
    ovms_connector.stub = FakeGrpcStub(None)

    result = ovms_connector.send(inference_input)
    assert "output" in result
    assert (result["output"] == expected_result["output"]).all()
