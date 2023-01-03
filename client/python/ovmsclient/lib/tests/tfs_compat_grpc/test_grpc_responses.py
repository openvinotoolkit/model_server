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

from numpy import ndarray
from numpy.core.numeric import array_equal
import pytest

from ovmsclient.tfs_compat.protos.tensorflow.core.framework.types_pb2 import DataType

from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.predict_pb2 import PredictResponse

from ovmsclient.tfs_compat.grpc.responses import (GrpcModelMetadataResponse,
                                                  GrpcModelStatusResponse,
                                                  GrpcPredictResponse)

from tfs_compat_grpc.config import (MODEL_METADATA_RESPONSE_VALID, MODEL_STATUS_RESPONSE_VALID,
                                    PREDICT_RESPONSE_VALID, PREDICT_RESPONSE_TENSOR_TYPE_INVALID)

from tfs_compat_grpc.utils import (create_model_metadata_response,
                                   create_model_status_response,
                                   merge_model_status_responses)


@pytest.mark.parametrize("outputs_dict, model_name, model_version,"
                         "expected_outputs", PREDICT_RESPONSE_VALID)
def test_PredictResponse_to_dict_valid(outputs_dict, model_name, model_version,
                                       expected_outputs):
    predict_raw_response = PredictResponse()

    predict_raw_response.model_spec.name = model_name
    predict_raw_response.model_spec.version.value = model_version

    for key, value in outputs_dict.items():
        predict_raw_response.outputs[key].CopyFrom(value)

    predict_response = GrpcPredictResponse(predict_raw_response)
    response_dict = predict_response.to_dict()

    assert isinstance(response_dict, dict)
    assert "outputs" in response_dict
    assert len(response_dict) == 1
    raw_response = predict_response.raw_response
    if isinstance(response_dict["outputs"], dict):
        for output_name, array in response_dict["outputs"].items():
            assert output_name in raw_response.outputs.keys()
            assert type(array) is ndarray
            assert array_equal(array, expected_outputs[output_name])
    else:
        assert type(response_dict["outputs"]) is ndarray
        assert array_equal(response_dict["outputs"], expected_outputs)


@pytest.mark.parametrize("outputs_dict, model_name, model_version, expected_exception,"
                         "expected_message", PREDICT_RESPONSE_TENSOR_TYPE_INVALID)
def test_PredictResponse_to_dict_invalid(outputs_dict, model_name, model_version,
                                         expected_exception, expected_message):
    predict_raw_response = PredictResponse()

    predict_raw_response.model_spec.name = model_name
    predict_raw_response.model_spec.version.value = model_version

    for key, value in outputs_dict.items():
        predict_raw_response.outputs[key].CopyFrom(value)

    predict_response = GrpcPredictResponse(predict_raw_response)
    with pytest.raises(expected_exception) as e_info:
        predict_response.to_dict()

    assert str(e_info.value) == expected_message


@pytest.mark.parametrize("model_raw_status_response_dict", MODEL_STATUS_RESPONSE_VALID)
def test_ModelStatusResponse_to_dict_valid(model_raw_status_response_dict):
    model_raw_responses = []
    for version, status in model_raw_status_response_dict.items():
        model_raw_responses.append(create_model_status_response(version,
                                   status['error_code'], status['error_message'],
                                   status['state']))
    raw_response = merge_model_status_responses(model_raw_responses)

    response = GrpcModelStatusResponse(raw_response)
    response_dict = response.to_dict()

    assert isinstance(response_dict, dict)
    assert len(response_dict) == len(model_raw_status_response_dict)
    for version, status in model_raw_status_response_dict.items():
        assert version in response_dict
        assert isinstance(response_dict[version], dict)
        assert response_dict[version]['error_code'] == status['error_code']
        assert response_dict[version]['error_message'] == status['error_message']
        assert response_dict[version]['state'] == ModelVersionStatus.State.Name(status['state'])


@pytest.mark.parametrize("model_raw_metadata_response_dict", MODEL_METADATA_RESPONSE_VALID)
def test_ModelMetadataResponse_to_dict_valid(model_raw_metadata_response_dict):
    raw_response = create_model_metadata_response(model_raw_metadata_response_dict)

    response = GrpcModelMetadataResponse(raw_response)
    response_dict = response.to_dict()

    assert isinstance(response_dict, dict)
    assert len(response_dict) == 3

    version = model_raw_metadata_response_dict['version']
    assert all(key in response_dict for key in ["model_version", "inputs", "outputs"])
    assert response_dict['model_version'] == version

    inputs = response_dict['inputs']
    assert len(inputs) == len(model_raw_metadata_response_dict['inputs'])
    for input in inputs:
        assert input in model_raw_metadata_response_dict['inputs']
        assert isinstance(inputs[input], dict)
        assert 'shape' in inputs[input] and 'dtype' in inputs[input]
        assert inputs[input]['shape'] == model_raw_metadata_response_dict['inputs'][input]['shape']
        assert (inputs[input]['dtype']
                == DataType.Name(model_raw_metadata_response_dict['inputs'][input]['dtype']))

    outputs = response_dict['outputs']
    assert len(outputs) == len(model_raw_metadata_response_dict['outputs'])
    for output in outputs:
        assert output in model_raw_metadata_response_dict['outputs']
        assert isinstance(outputs[output], dict)
        assert 'shape' in outputs[output] and 'dtype' in outputs[output]
        assert (outputs[output]['shape']
                == model_raw_metadata_response_dict['outputs'][output]['shape'])
        assert (outputs[output]['dtype']
                == DataType.Name(model_raw_metadata_response_dict['outputs'][output]['dtype']))
