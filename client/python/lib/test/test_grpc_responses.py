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

from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef, TensorInfo
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataResponse, SignatureDefMap
from tensorflow_serving.apis.model_pb2 import ModelSpec
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusResponse, ModelVersionStatus
from tensorflow_serving.util.status_pb2 import StatusProto

from google.protobuf.any_pb2 import Any

from ovmsclient.tfs_compat.grpc.responses import GrpcModelMetadataResponse, GrpcModelStatusResponse
from config import MODEL_METADATA_RESPONSE_VALID, MODEL_STATUS_RESPONSE_VALID

@pytest.mark.parametrize("model_raw_status_response_dict" , MODEL_STATUS_RESPONSE_VALID)
def test_ModelStatusResponse_to_dict_valid(model_raw_status_response_dict):
    model_raw_responses = []
    for version, status in model_raw_status_response_dict.items():
        model_raw_responses.append(create_model_status_response(version, status['error_code'], status['error_message'],
         status['state']))
    raw_response = merge_model_status_responses(model_raw_responses)

    response = GrpcModelStatusResponse(raw_response)
    response_dictionary = response.to_dict()
    assert isinstance(response_dictionary, dict)
    assert len(response_dictionary) == len(model_raw_status_response_dict)
    for version, status in model_raw_status_response_dict.items():
        assert version in response_dictionary
        assert isinstance(response_dictionary[version], dict)
        assert response_dictionary[version]['error_code'] == status['error_code']
        assert response_dictionary[version]['error_message'] == status['error_message']
        assert response_dictionary[version]['state'] == ModelVersionStatus.State.Name(status['state'])

@pytest.mark.parametrize("model_raw_metadata_response_dict" , MODEL_METADATA_RESPONSE_VALID)
def test_ModelMetadataResponse_to_dict_valid(model_raw_metadata_response_dict):
    raw_response = create_model_metadata_response(model_raw_metadata_response_dict)

    response = GrpcModelMetadataResponse(raw_response)
    response_dictionary = response.to_dict()
    assert isinstance(response_dictionary, dict)
    assert len(response_dictionary) == 1

    version = model_raw_metadata_response_dict['version']
    assert version in response_dictionary
    assert 'inputs' in response_dictionary[version] and 'outputs' in response_dictionary[version]

    inputs = response_dictionary[version]['inputs']
    assert len(inputs) == len(model_raw_metadata_response_dict['inputs'])
    for input in inputs:
        assert input in model_raw_metadata_response_dict['inputs']
        assert isinstance(inputs[input], dict)
        assert 'shape' in inputs[input] and 'dtype' in inputs[input]
        assert inputs[input]['shape'] == model_raw_metadata_response_dict['inputs'][input]['shape']
        assert inputs[input]['dtype'] == GrpcModelMetadataResponse._DTYPE_NAME[
            model_raw_metadata_response_dict['inputs'][input]['dtype']
        ]


    outputs = response_dictionary[version]['outputs']
    assert len(outputs) == len(model_raw_metadata_response_dict['outputs'])
    for output in outputs:
        assert output in model_raw_metadata_response_dict['outputs']
        assert isinstance(outputs[output], dict)
        assert 'shape' in outputs[output] and 'dtype' in outputs[output]
        assert outputs[output]['shape'] == model_raw_metadata_response_dict['outputs'][output]['shape']
        assert outputs[output]['dtype'] == GrpcModelMetadataResponse._DTYPE_NAME[
            model_raw_metadata_response_dict['outputs'][output]['dtype']
        ]

def create_model_metadata_response(model_raw_metadata_response_dict):
    raw_response = GetModelMetadataResponse()

    model_spec = ModelSpec()
    model_spec.name = model_raw_metadata_response_dict['name']
    model_spec.version.value = model_raw_metadata_response_dict['version']

    raw_response.model_spec.CopyFrom(model_spec)

    metadata = SignatureDef()
    
    inputs = model_raw_metadata_response_dict['inputs']
    for input_key in inputs.keys():
        input_info = inputs[input_key]

        input_tensor_info = TensorInfo()
        input_tensor_info.name = input_key
        input_tensor_info.dtype = input_info['dtype']

        input_tensor_shape = TensorShapeProto(dim=[
            TensorShapeProto.Dim(size=x) for x in input_info['shape']
        ])
        input_tensor_info.tensor_shape.CopyFrom(input_tensor_shape)

        metadata.inputs[input_key].CopyFrom(input_tensor_info)

    outputs = model_raw_metadata_response_dict['outputs']
    for output_key in outputs.keys():
        output_info = outputs[output_key]

        output_tensor_info = TensorInfo()
        output_tensor_info.name = output_key
        output_tensor_info.dtype = output_info['dtype']

        output_tensor_shape = TensorShapeProto(dim=[
            TensorShapeProto.Dim(size=x) for x in output_info['shape']
        ])
        output_tensor_info.tensor_shape.CopyFrom(output_tensor_shape)

        metadata.outputs[output_key].CopyFrom(output_tensor_info)

    signature_def_map = SignatureDefMap()
    signature_def_map.signature_def['serving_default'].CopyFrom(metadata)
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
