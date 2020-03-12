#
# Copyright (c) 2019-2020 Intel Corporation
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

from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2, get_model_metadata_pb2, \
    get_model_status_pb2


def infer(img, input_tensor, grpc_stub, model_spec_name,
          model_spec_version, output_tensors):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    if model_spec_version is not None:
        request.model_spec.version.value = model_spec_version
    print("input shape ", img.shape)
    request.inputs[input_tensor].CopyFrom(
        make_tensor_proto(img, shape=list(img.shape)))
    result = grpc_stub.Predict(request, 10.0)
    data = {}
    for output_tensor in output_tensors:
        data[output_tensor] = make_ndarray(result.outputs[output_tensor])
    return data


def get_model_metadata(model_name, metadata_field: str = "signature_def",
                       version=None):
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = model_name
    if version is not None:
        request.model_spec.version.value = version
    request.metadata_field.append(metadata_field)
    return request


def model_metadata_response(response):
    signature_def = response.metadata['signature_def']
    signature_map = get_model_metadata_pb2.SignatureDefMap()
    signature_map.ParseFromString(signature_def.value)
    serving_default = signature_map.ListFields()[0][1]['serving_default']
    serving_inputs = serving_default.inputs
    input_blobs_keys = {key: {} for key in serving_inputs.keys()}
    tensor_shape = {key: serving_inputs[key].tensor_shape
                    for key in serving_inputs.keys()}
    for input_blob in input_blobs_keys:
        inputs_shape = [d.size for d in tensor_shape[input_blob].dim]
        tensor_dtype = serving_inputs[input_blob].dtype
        input_blobs_keys[input_blob].update({'shape': inputs_shape})
        input_blobs_keys[input_blob].update({'dtype': tensor_dtype})

    serving_outputs = serving_default.outputs
    output_blobs_keys = {key: {} for key in serving_outputs.keys()}
    tensor_shape = {key: serving_outputs[key].tensor_shape
                    for key in serving_outputs.keys()}
    for output_blob in output_blobs_keys:
        outputs_shape = [d.size for d in tensor_shape[output_blob].dim]
        tensor_dtype = serving_outputs[output_blob].dtype
        output_blobs_keys[output_blob].update({'shape': outputs_shape})
        output_blobs_keys[output_blob].update({'dtype': tensor_dtype})

    return input_blobs_keys, output_blobs_keys


def get_model_status(model_name, version=None):
    request = get_model_status_pb2.GetModelStatusRequest()
    request.model_spec.name = model_name
    if version is not None:
        request.model_spec.version.value = version
    return request
