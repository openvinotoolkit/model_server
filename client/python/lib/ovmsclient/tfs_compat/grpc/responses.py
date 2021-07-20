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

from tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus

from ovmsclient.tfs_compat.base.responses import PredictResponse, ModelMetadataResponse, ModelStatusResponse
from tensorflow_serving.apis import get_model_metadata_pb2

class GrpcPredictResponse(PredictResponse):
    pass

class GrpcModelMetadataResponse(ModelMetadataResponse):
    
    _DTYPE_NAME = [ 'DT_INVALID',
               'DT_FLOAT',
               'DT_DOUBLE',
               'DT_INT32',
               'DT_UINT8',
               'DT_INT16',
               'DT_INT8',
               'DT_STRING',
               'DT_COMPLEX64',
               'DT_INT64',
               'DT_BOOL',
               'DT_QINT8',
               'DT_QUINT8',
               'DT_QINT32',
               'DT_BFLOAT16',
               'DT_QINT16',
               'DT_QUINT16',
               'DT_UINT16',
               'DT_COMPLEX128',
               'DT_HALF',
               'DT_RESOURCE',
               'DT_VARIANT',
               'DT_UINT32',
               'DT_UINT64']

    def to_dict(self):
        result_dictionary = {}

        signature_def = self.raw_response.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)
        serving_default = signature_map.ListFields()[0][1]['serving_default']
        
        serving_inputs = serving_default.inputs
        input_blobs = {key : {} for key in serving_inputs.keys()}
        tensor_shape = {key: serving_inputs[key].tensor_shape for key in serving_inputs.keys()}

        input_blobs_dictionary = {}
        for input_blob in input_blobs:
            inputs_shape = [d.size for d in tensor_shape[input_blob].dim]
            tensor_dtype = serving_inputs[input_blob].dtype
            input_blobs_dictionary[input_blob] = dict([
                ("shape", inputs_shape),
                ("dtype", self._DTYPE_NAME[tensor_dtype])
            ])
        
        serving_outputs = serving_default.outputs
        output_blobs = {key: {} for key in serving_outputs.keys()}
        tensor_shape = {key: serving_outputs[key].tensor_shape for key in serving_outputs.keys()}

        output_blobs_dictionary = {}
        for output_blob in output_blobs:
            outputs_shape = [d.size for d in tensor_shape[output_blob].dim]
            tensor_dtype = serving_outputs[output_blob].dtype
            output_blobs_dictionary[output_blob] = dict([
                ("shape", outputs_shape),
                ("dtype", self._DTYPE_NAME[tensor_dtype])
            ])
    
        version = self.raw_response.model_spec.version.value
        result_dictionary[version] = dict([
            ("inputs", input_blobs_dictionary),
            ("outputs", output_blobs_dictionary)
        ])
        return result_dictionary

class GrpcModelStatusResponse(ModelStatusResponse):

    def to_dict(self):
        result_dictionary = {}
        model_version_status = self.raw_response.model_version_status
        for model_version in model_version_status:
            result_dictionary[model_version.version] = dict([
                ('state', ModelVersionStatus.State.Name(model_version.state)),
                ('error_code', model_version.status.error_code),
                ('error_message', model_version.status.error_message),
            ])
        return result_dictionary
