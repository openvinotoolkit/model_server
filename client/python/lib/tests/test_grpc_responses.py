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

from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef, TensorInfo
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.core.framework.types_pb2 import DataType

from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataResponse, SignatureDefMap
from tensorflow_serving.apis.model_pb2 import ModelSpec
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusResponse, ModelVersionStatus
from tensorflow_serving.apis.predict_pb2 import PredictResponse
from tensorflow_serving.util.status_pb2 import StatusProto

from google.protobuf.any_pb2 import Any

from ovmsclient.tfs_compat.grpc.responses import GrpcModelMetadataResponse, GrpcModelStatusResponse, GrpcPredictResponse

from config import MODEL_METADATA_RESPONSE_VALID, MODEL_STATUS_RESPONSE_VALID, PREDICT_RESPONSE_VALID, PREDICT_RESPONSE_INVALID

@pytest.mark.parametrize("outputs_dict, model_name, model_version, expected_outputs_dict", PREDICT_RESPONSE_VALID)
def test_PredictResponse_to_dict_valid(outputs_dict, model_name, model_version, expected_outputs_dict):
    predict_raw_response = PredictResponse()

    predict_raw_response.model_spec.name = model_name
    predict_raw_response.model_spec.version.value = model_version

    for key, value in outputs_dict.items():
        predict_raw_response.outputs[key].CopyFrom(value)

    predict_response = GrpcPredictResponse(predict_raw_response)
    response_dict = predict_response.to_dict()

    assert isinstance(response_dict, dict)
    assert len(response_dict) == len(outputs_dict)
    raw_response = predict_response.raw_response
    for output_name, array in response_dict.items():
        assert output_name in raw_response.outputs.keys()
        assert type(array) is ndarray
        assert array_equal(array, expected_outputs_dict[output_name])

@pytest.mark.parametrize("outputs_dict, model_name, model_version, expected_exception, expected_message", PREDICT_RESPONSE_INVALID)
def test_PredictResponse_to_dict_invalid(outputs_dict, model_name, model_version, expected_exception, expected_message):
    predict_raw_response = PredictResponse()

    predict_raw_response.model_spec.name = model_name
    predict_raw_response.model_spec.version.value = model_version

    for key, value in outputs_dict.items():
        predict_raw_response.outputs[key].CopyFrom(value)

    predict_response = GrpcPredictResponse(predict_raw_response)
    with pytest.raises(expected_exception) as e_info:
        response_dict = predict_response.to_dict()
    
    assert str(e_info.value) == expected_message

@pytest.mark.parametrize("model_raw_status_response_dict" , MODEL_STATUS_RESPONSE_VALID)
def test_ModelStatusResponse_to_dict_valid(model_raw_status_response_dict):
    model_raw_responses = []
    for version, status in model_raw_status_response_dict.items():
        model_raw_responses.append(create_model_status_response(version, status['error_code'], status['error_message'],
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

@pytest.mark.parametrize("model_raw_metadata_response_dict" , MODEL_METADATA_RESPONSE_VALID)
def test_ModelMetadataResponse_to_dict_valid(model_raw_metadata_response_dict):
    raw_response = create_model_metadata_response(model_raw_metadata_response_dict)

    response = GrpcModelMetadataResponse(raw_response)
    response_dict = response.to_dict()
    
    assert isinstance(response_dict, dict)
    assert len(response_dict) == 1

    version = model_raw_metadata_response_dict['version']
    assert version in response_dict
    assert 'inputs' in response_dict[version] and 'outputs' in response_dict[version]

    inputs = response_dict[version]['inputs']
    assert len(inputs) == len(model_raw_metadata_response_dict['inputs'])
    for input in inputs:
        assert input in model_raw_metadata_response_dict['inputs']
        assert isinstance(inputs[input], dict)
        assert 'shape' in inputs[input] and 'dtype' in inputs[input]
        assert inputs[input]['shape'] == model_raw_metadata_response_dict['inputs'][input]['shape']
        assert inputs[input]['dtype'] == DataType.Name(model_raw_metadata_response_dict['inputs'][input]['dtype'])


    outputs = response_dict[version]['outputs']
    assert len(outputs) == len(model_raw_metadata_response_dict['outputs'])
    for output in outputs:
        assert output in model_raw_metadata_response_dict['outputs']
        assert isinstance(outputs[output], dict)
        assert 'shape' in outputs[output] and 'dtype' in outputs[output]
        assert outputs[output]['shape'] == model_raw_metadata_response_dict['outputs'][output]['shape']
        assert outputs[output]['dtype'] == DataType.Name(model_raw_metadata_response_dict['outputs'][output]['dtype'])

def create_model_metadata_response(model_raw_metadata_response_dict):
    raw_response = GetModelMetadataResponse()

    model_spec = ModelSpec()
    model_spec.name = model_raw_metadata_response_dict['name']
    model_spec.version.value = model_raw_metadata_response_dict['version']

    raw_response.model_spec.CopyFrom(model_spec)

    signature_def = SignatureDef()
    
    inputs = model_raw_metadata_response_dict['inputs']
    for input_name, input_info in inputs.items():
        input_tensor_info = TensorInfo()
        input_tensor_info.name = input_name
        input_tensor_info.dtype = input_info['dtype']

        input_tensor_shape = TensorShapeProto(dim=[
            TensorShapeProto.Dim(size=x) for x in input_info['shape']
        ])
        input_tensor_info.tensor_shape.CopyFrom(input_tensor_shape)

        signature_def.inputs[input_name].CopyFrom(input_tensor_info)

    outputs = model_raw_metadata_response_dict['outputs']
    for output_name, output_info in outputs.items():
        output_tensor_info = TensorInfo()
        output_tensor_info.name = output_name
        output_tensor_info.dtype = output_info['dtype']

        output_tensor_shape = TensorShapeProto(dim=[
            TensorShapeProto.Dim(size=x) for x in output_info['shape']
        ])
        output_tensor_info.tensor_shape.CopyFrom(output_tensor_shape)

        signature_def.outputs[output_name].CopyFrom(output_tensor_info)

    signature_def_map = SignatureDefMap()
    signature_def_map.signature_def['serving_default'].CopyFrom(signature_def)
    serialized_metadata = signature_def_map.SerializeToString()

    metadata = Any()
    metadata.type_url = "type.googleapis.com/tensorflow.serving.SignatureDefMap"
    metadata.value = serialized_metadata

    raw_response.metadata['signature_def'].CopyFrom(metadata)

    return raw_response

def create_model_status_response(model_version, error_code, error_message, model_state):
    status = StatusProto()
    status.error_code = error_code
    status.error_message = error_message

    model_version_status = ModelVersionStatus()
    model_version_status.version = model_version
    model_version_status.state = model_state
    model_version_status.status.CopyFrom(status)

    return model_version_status

def merge_model_status_responses(responses):
    raw_response = GetModelStatusResponse()
    model_versions = [response for response in responses]
    raw_response.model_version_status.extend(model_versions)

    return raw_response
