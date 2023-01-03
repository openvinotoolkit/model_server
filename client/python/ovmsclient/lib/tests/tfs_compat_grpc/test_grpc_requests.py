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

import pytest

from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_pb2 import TensorProto
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest # noqa

from ovmsclient.tfs_compat.grpc.requests import (GrpcModelMetadataRequest, GrpcPredictRequest,
                                                 make_metadata_request, make_predict_request,
                                                 make_status_request, GrpcModelStatusRequest)

from tfs_compat_grpc.config import (PREDICT_REQUEST_INVALID_INPUTS, PREDICT_REQUEST_VALID)
from config import (MODEL_SPEC_INVALID, MODEL_SPEC_VALID)

from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.predict_pb2 import PredictRequest


@pytest.mark.parametrize("name, version", MODEL_SPEC_VALID)
def test_make_status_request_valid(mocker, name, version):
    mock_method = mocker.patch('ovmsclient.tfs_compat.grpc.requests._check_model_spec')
    model_status_request = make_status_request(name, version)

    mock_method.assert_called_once()
    assert isinstance(model_status_request, GrpcModelStatusRequest)
    assert model_status_request.model_version == version
    assert model_status_request.model_name == name
    assert isinstance(model_status_request.raw_request, GetModelStatusRequest)


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_make_status_request_invalid(mocker, name, version, expected_exception, expected_message):
    mock_method = mocker.patch('ovmsclient.tfs_compat.grpc.requests._check_model_spec',
                               side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        make_status_request(name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


@pytest.mark.parametrize("name, version", MODEL_SPEC_VALID)
def test_make_metadata_request_vaild(mocker, name, version):
    mock_method = mocker.patch('ovmsclient.tfs_compat.grpc.requests._check_model_spec')
    model_metadata_request = make_metadata_request(name, version)

    mock_method.assert_called_once()
    assert isinstance(model_metadata_request, GrpcModelMetadataRequest)
    assert model_metadata_request.model_version == version
    assert model_metadata_request.model_name == name
    assert isinstance(model_metadata_request.raw_request, GetModelMetadataRequest)
    assert len(model_metadata_request.raw_request.metadata_field) == 1
    assert model_metadata_request.raw_request.metadata_field[0] == 'signature_def'


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_make_metadata_request_invalid(mocker, name, version, expected_exception, expected_message):
    mock_method = mocker.patch('ovmsclient.tfs_compat.grpc.requests._check_model_spec',
                               side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        make_metadata_request(name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


@pytest.mark.parametrize("inputs, expected_proto, name, version", PREDICT_REQUEST_VALID)
def test_make_predict_request_valid(inputs, expected_proto, name, version):
    model_predict_request = make_predict_request(inputs, name, version)

    raw_predict_request = model_predict_request.raw_request

    assert isinstance(model_predict_request, GrpcPredictRequest)
    assert model_predict_request.model_name == name
    assert model_predict_request.model_version == version
    assert model_predict_request.inputs == inputs
    assert isinstance(raw_predict_request, PredictRequest)
    assert raw_predict_request.model_spec.name == name
    assert raw_predict_request.model_spec.version.value == version
    assert len(inputs.keys()) == len(list(raw_predict_request.inputs.keys()))
    for key, value in inputs.items():
        assert isinstance(raw_predict_request.inputs[key], TensorProto)
        if isinstance(value, TensorProto):
            assert value == raw_predict_request.inputs[key]
        else:
            assert (raw_predict_request.inputs[key].__getattribute__(expected_proto[key]['field'])
                    == expected_proto[key]['value'])
            assert raw_predict_request.inputs[key].tensor_shape == expected_proto[key]['shape']
            assert raw_predict_request.inputs[key].dtype == expected_proto[key]['dtype']


@pytest.mark.parametrize("name, version, expected_exception, expected_message", MODEL_SPEC_INVALID)
def test_make_predict_request_invalid_model_spec(mocker, name, version,
                                                 expected_exception, expected_message):
    inputs = {
        "input": [1, 2, 3]
    }
    mock_method = mocker.patch('ovmsclient.tfs_compat.grpc.requests._check_model_spec',
                               side_effect=expected_exception(expected_message))
    with pytest.raises(expected_exception) as e_info:
        make_predict_request(inputs, name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()


@pytest.mark.causes_deprecation_warning(triggered_by="numpy<1.24")
@pytest.mark.parametrize("""inputs, name, version,
                            expected_exception, expected_message""", PREDICT_REQUEST_INVALID_INPUTS)
def test_make_predict_request_invalid_inputs(mocker, inputs, name, version,
                                             expected_exception, expected_message):
    mock_method = mocker.patch('ovmsclient.tfs_compat.grpc.requests._check_model_spec')
    with pytest.raises(expected_exception) as e_info:
        make_predict_request(inputs, name, version)

    assert str(e_info.value) == expected_message
    mock_method.assert_called_once()
