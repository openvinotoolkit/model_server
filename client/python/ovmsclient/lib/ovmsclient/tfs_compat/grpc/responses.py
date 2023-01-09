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

from ovmsclient.tfs_compat.protos.tensorflow_serving.apis.get_model_status_pb2 import ModelVersionStatus # noqa
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis import get_model_metadata_pb2
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.types_pb2 import DataType

from ovmsclient.tfs_compat.base.responses import (PredictResponse, ModelMetadataResponse,
                                                  ModelStatusResponse)
from ovmsclient.tfs_compat.grpc.tensors import make_ndarray


class GrpcPredictResponse(PredictResponse):

    def to_dict(self):
        result_dict = {"outputs": {}}
        output_names = list(self.raw_response.outputs.keys())

        for key, value in self.raw_response.outputs.items():
            result_dict["outputs"][key] = make_ndarray(value)
        # For models with only one output, put results directly under "outputs" key
        if len(output_names) == 1:
            result_dict["outputs"] = result_dict["outputs"][output_names[0]]

        return result_dict


class GrpcModelMetadataResponse(ModelMetadataResponse):

    def to_dict(self):

        signature_def = self.raw_response.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)
        model_signature = signature_map.ListFields()[0][1]['serving_default']

        inputs_metadata = {}
        for input_name, input_info in model_signature.inputs.items():
            input_shape = [d.size for d in input_info.tensor_shape.dim]
            inputs_metadata[input_name] = dict([
                ("shape", input_shape),
                ("dtype", DataType.Name(input_info.dtype))
            ])

        outputs_metadata = {}
        for output_name, output_info in model_signature.outputs.items():
            output_shape = [d.size for d in output_info.tensor_shape.dim]
            outputs_metadata[output_name] = dict([
                ("shape", output_shape),
                ("dtype", DataType.Name(output_info.dtype))
            ])

        version = self.raw_response.model_spec.version.value
        result_dict = dict([
            ("model_version", version),
            ("inputs", inputs_metadata),
            ("outputs", outputs_metadata)
        ])
        return result_dict


class GrpcModelStatusResponse(ModelStatusResponse):

    def to_dict(self):
        result_dict = {}
        model_version_status = self.raw_response.model_version_status
        for model_version in model_version_status:
            result_dict[model_version.version] = dict([
                ('state', ModelVersionStatus.State.Name(model_version.state)),
                ('error_code', model_version.status.error_code),
                ('error_message', model_version.status.error_message),
            ])
        return result_dict
