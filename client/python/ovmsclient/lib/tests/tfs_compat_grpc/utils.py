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

from ovmsclient.tfs_compat.protos.tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef, TensorInfo # noqa
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataResponse, SignatureDefMap # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.model_pb2 import ModelSpec
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusResponse, ModelVersionStatus # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.util.status_pb2 import StatusProto

from google.protobuf.any_pb2 import Any


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
