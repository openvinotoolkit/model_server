#
# Copyright (c) 2026 Intel Corporation
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

import json

import numpy as np
import tensorflow
from google.protobuf.json_format import MessageToJson, Parse
from tensorboard.util.tensor_util import make_ndarray
from tensorflow import make_tensor_proto
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import (
    get_model_metadata_pb2, get_model_status_pb2, model_service_pb2_grpc, predict_pb2, prediction_service_pb2_grpc,)

from tests.functional.utils.assertions import InvalidMetadataException, NotSupported
from common_libs.http.base import HttpMethod
from tests.functional.utils.inference.communication.grpc import GRPC_TIMEOUT
from tests.functional.utils.inference.serving.base import AbstractServingWrapper
from tests.functional.utils.logger import get_logger
from ovms.constants.metrics import Metric
from tests.functional.constants.ovms import Ovms

logger = get_logger(__name__)

TFS = "TFS"


class TensorFlowServingWrapper(AbstractServingWrapper):
    REST_VERSION = "v1"
    PREDICT = ":predict"

    METRICS_PROTOCOL = Metric.TensorFlowServing


    def set_grpc_stubs(self):
        """
            Assigns objects for inference purposes.
        """
        self.predict_stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.model_service_stub = model_service_pb2_grpc.ModelServiceStub(self.channel)

    def get_model_status_grpc_request(self, model_name=None, version=None):
        request = get_model_status_pb2.GetModelStatusRequest()
        request.model_spec.name = self.model_name if model_name is None else model_name
        if version is not None:
            request.model_spec.version.value = int(version)

        if self.model_version is not None:
            request.model_spec.version.value = int(self.model_version)
        return request

    def get_model_meta_grpc_request(self, model_name=None):
        metadata_field = "signature_def"
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = self.model_name
        if self.model_version is not None:
            request.model_spec.version.value = int(self.model_version)
        request.metadata_field.append(metadata_field)
        return request

    def send_model_status_grpc_request(self, request):
        response = self.model_service_stub.GetModelStatus(
            request, wait_for_ready=True, timeout=GRPC_TIMEOUT
        )
        return response

    def send_model_meta_grpc_request(self, request):
        response = self.predict_stub.GetModelMetadata(
            request, wait_for_ready=True, timeout=GRPC_TIMEOUT
        )
        return response

    def get_predict_grpc_request(self, input_objects, raw=False, mediapipe_name=None):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        if self.model_version is not None:
            request.model_spec.version.value = int(self.model_version)
        for input_name, input_object in input_objects.items():
            request.inputs[input_name].CopyFrom(
                make_tensor_proto(input_object, shape=input_object.shape)
            )
        return request

    @staticmethod
    def process_predict_grpc_output(result, **kwargs):
        outputs = {
            output_name: make_ndarray(output) for output_name, output in result.outputs.items()
        }
        return outputs

    def create_inference(self):
        """
            Assigns objects for inference purposes.
        """
        # method from brother class (multiple inheritance)
        self.communication_service = self.create_communication_service()
        return self.communication_service

    def send_predict_grpc_request(self, request, timeout=GRPC_TIMEOUT):
        return self.predict_stub.Predict(request, wait_for_ready=True, timeout=timeout)

    def predict(self, request, timeout=60, raw=False):
        result = self.send_predict_request(request, timeout)
        outputs = self.process_predict_output(result)
        return outputs

    def get_rest_path(self, operation=None, model_version=None, model_name=None):
        """
            Expect 2 REST path formats for TF format:
             - GET: (METADATA, MODELS)
                http://{REST_URL}:{REST_PORT}/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}/{OPERATION}
             - POST: (PREDICT)
                http://{REST_URL}:{REST_PORT}/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}:predict
        """
        model_name = model_name if model_name is not None else (self.model.name if self.model else self.model_name)
        assert model_name
        rest_path = [self.REST_VERSION, self.MODELS, model_name]
        if model_version is not None:
            rest_path.append(self.VERSIONS)
            rest_path.append(str(model_version))
        if operation == self.PREDICT:
            rest_path[-1] = "".join([rest_path[-1], operation])
        elif operation not in [self.STATUS, None]:
            rest_path.append(operation)
        rest_path = "/".join(rest_path)
        return rest_path


    @staticmethod
    def prepare_body_dict(input_objects: dict, request_format=Ovms.BINARY_IO_LAYOUT_ROW_NAME, **kwargs):
        """
            Prepare HTTP request's body in given format:
                - row_name,
                - column_name,
                - row_noname,
                - column_noname
        """
        signature = "serving_default"
        if request_format == Ovms.BINARY_IO_LAYOUT_ROW_NAME:
            instances = []
            for input_name, input_object in input_objects.items():
                if input_object.shape:
                    for i in range(0, input_object.shape[0], 1):
                        input_data = input_object[i].decode() if (
                                input_object.dtype == np.object_) else input_object[i].tolist()
                        instances.append({input_name: input_data})
                else:
                    instances.append({input_name: str(input_object[()])})
                    # https://numpy.org/doc/stable/reference/arrays.scalars.html#indexing
            data_obj = {"signature_name": signature, "instances": instances}
        elif request_format == Ovms.BINARY_IO_LAYOUT_ROW_NONAME:
            instances = []
            for input_object in input_objects.values():
                if input_object.shape:
                    for i in range(0, input_object.shape[0]):
                        input_data = input_object[i].decode() if (
                                input_object.dtype == np.object_) else input_object[i].tolist()
                        instances.append(input_data)
                else:
                    # https://numpy.org/doc/stable/reference/arrays.scalars.html#indexing
                    instances.append([str(input_object[()])])
            data_obj = {"signature_name": signature, 'instances': instances}
        elif request_format == Ovms.BINARY_IO_LAYOUT_COLUMN_NAME:
            inputs = {}
            for input_name, input_object in input_objects.items():
                inputs[input_name] = [x.decode() for x in input_object.tolist()] if input_object.dtype == np.object_ \
                    else input_object.tolist()
            data_obj = {"signature_name": signature, 'inputs': inputs}
        elif request_format == Ovms.BINARY_IO_LAYOUT_COLUMN_NONAME:
            assert len(input_objects) == 1, \
                f"Only single input is required if {Ovms.BINARY_IO_LAYOUT_COLUMN_NONAME} format is used"
            input_object = list(input_objects.items())[0][1]
            _input = [x.decode() for x in input_object.tolist()] if input_object.dtype == np.object_ \
                else input_object.tolist()
            data_obj = {"signature_name": signature, 'inputs': _input}
        else:
            raise ValueError(f"Unknown response format: {request_format}")
        return data_obj

    def get_inputs_outputs_from_response(self, response):
        # expect content to be dictionary encoded as bytes:
        # model.content == b'{\n "modelSpec": {\n  "name": "resnet-50-tf",\n  "signatureName": "" ...
        model_specification = json.loads(response.text)

        serving_default = model_specification['metadata']['signature_def']['signatureDef']['serving_default']

        self.model.inputs = {}
        self.model.outputs = {}

        for name, details in serving_default['inputs'].items():
            self.model.inputs[details['name']] = {
                'shape': [int(x['size']) for x in details["tensorShape"]["dim"]],
                'dtype': tensorflow.dtypes.as_dtype(getattr(types_pb2, details['dtype']))
            }

        for name, details in serving_default['outputs'].items():
            self.model.outputs[details['name']] = {
                'shape': [int(x['size']) for x in details["tensorShape"]["dim"]],
                'dtype': tensorflow.dtypes.as_dtype(getattr(types_pb2, details['dtype']))
            }

    def process_json_output(self, result_dict):
        """
            Converts predict result to output as a numpy array.
            Input:
                result_dict = {'predictions': []}
            Output:
                <output_name> = {ndarray: (1, 1001)}
        """
        output = {}
        if "outputs" in result_dict:
            key_name = "outputs"
            if isinstance(result_dict[key_name], dict):
                for output_tensor in self.output_names:
                    output[output_tensor] = np.asarray(result_dict[key_name][output_tensor])
            else:
                output[self.output_names[0]] = np.asarray(result_dict[key_name])
        elif "predictions" in result_dict:
            key_name = "predictions"
            if isinstance(result_dict[key_name][0], dict):
                for row in result_dict[key_name]:
                    for output_tensor in self.output_names:
                        if output_tensor not in output:
                            output[output_tensor] = []
                        output[output_tensor].append(row[output_tensor])
                for output_tensor in self.output_names:
                    output[output_tensor] = np.asarray(output[output_tensor])
            else:
                output[self.output_names[0]] = np.asarray(result_dict[key_name])
        else:
            logger.error(f"Missing required response in {result_dict}")
        return output

    def set_serving_inputs_outputs_grpc(self, response, **kwargs):
        """
            Sets inference response inputs and outputs.
            Parameters:
                response (GetModelMetadataResponse): inference response
        """
        signature_def = response.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)
        serving_default = signature_map.ListFields()[0][1]['serving_default']

        inputs = {}
        outputs = {}

        for name, details in serving_default.inputs.items():
            inputs[name] = {
                'shape': [x.size for x in details.tensor_shape.dim],
                'dtype': tensorflow.dtypes.as_dtype(details.dtype)
            }

        for name, details in serving_default.outputs.items():
            outputs[name] = {
                'shape': [x.size for x in details.tensor_shape.dim],
                'dtype': tensorflow.dtypes.as_dtype(details.dtype)
            }

        self.set_inputs(inputs)
        self.set_outputs(outputs)


    @staticmethod
    def get_data_type(data_type):
        """
            Converts given data_type to numpy format.
            Parameters:
                data_type (int)
            Returns:
                result (np)
        """
        result = None
        if data_type == 6:
            result = np.int8
        elif data_type == 3:
            result = np.int32
        elif data_type == 9:
            result = np.int64
        elif data_type == 1:
            result = np.float32
        else:
            raise NotImplementedError()
        return result

    def get_model_status_rest(self, timeout=60, version=None, model_name=None):
        rest_path = self.get_rest_path(None, model_version=version, model_name=model_name)
        response = self.client.request(HttpMethod.GET, path=rest_path, timeout=timeout, raw_response=True)

        # Transform JSON friendly output to protobuf compatible object required by callers.
        status_pb = get_model_status_pb2.GetModelStatusResponse()
        response = Parse(response.text, status_pb, ignore_unknown_fields=False)

        return response

    def prepare_stateful_request_rest(self, input_objects: dict, sequence_ctrl=None, sequence_id=None,
                                      ctrl_dtype=None, id_dtype=None):
        data_obj = self.prepare_body_dict(
            input_objects, request_format=Ovms.BINARY_IO_LAYOUT_COLUMN_NAME
        )
        if sequence_ctrl is not None:
            data_obj['inputs']['sequence_control_input'] = [sequence_ctrl]
        if sequence_id is not None:
            data_obj['inputs']['sequence_id'] = [sequence_id]
        return {'request': json.dumps(data_obj)}

    def predict_stateful_request_rest(self, request, timeout=900):
        result = self.send_predict_request(request, timeout)

        output_json = json.loads(result.text)
        sequence_id = output_json['outputs'].pop('sequence_id')[0] \
            if 'outputs' in output_json else None
        outputs = self.process_json_output(output_json)
        return sequence_id, outputs

    def prepare_stateful_request_grpc(self, input_objects: dict, sequence_ctrl=None, sequence_id=None,
                                      ctrl_dtype=None, id_dtype=None):
        sequence_id_dtype = id_dtype if id_dtype else 'uint64'
        sequence_ctrl_dtype = ctrl_dtype if ctrl_dtype else 'uint32'
        request = self.prepare_request(input_objects)
        if sequence_ctrl is not None:
            request['request'].inputs['sequence_control_input'].CopyFrom(
                make_tensor_proto([sequence_ctrl], dtype=sequence_ctrl_dtype)
            )
        if sequence_id is not None:
            request['request'].inputs['sequence_id'].CopyFrom(
                make_tensor_proto([sequence_id], dtype=sequence_id_dtype)
            )
        return request

    def predict_stateful_request_grpc(self, request, timeout=900):
        result = self.predict_stub.Predict(
            request, wait_for_ready=True, timeout=timeout
        )
        data = {}
        sequence_id = result.outputs.pop('sequence_id').uint64_val[0]
        for output_name, output in result.outputs.items():
            data[output_name] = make_ndarray(output)
        return sequence_id, data


    # KFS not supported API calls:
    def is_server_live_grpc(self):
        raise NotSupported("is_server_live is not available in TFS")

    def is_server_live_rest(self):
        raise NotSupported("is_server_live is not available in TFS")

    def is_server_ready_grpc(self):
        raise NotSupported("is_server_live is not available in TFS")

    def is_server_ready_rest(self):
        raise NotSupported("is_server_live is not available in TFS")

    def is_model_ready_grpc(self, model_name, model_version=""):
        """
           Gets information about model readiness (specific only for KFS - gRPC or REST).
           GET http://${REST_URL}:${REST_PORT}/v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready
           Response: True (ready) or False (not ready)
        """
        raise NotSupported()

    def is_model_ready_rest(self, model_name, model_version=""):
        """
           Gets information about model readiness (specific only for KFS - gRPC or REST).
           GET http://${REST_URL}:${REST_PORT}/v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready
           Response: True (ready) or False (not ready)
        """
        raise NotSupported()

    def cast_type_to_string(self, data_type):
        # https://github.com/openvinotoolkit/model_server/blob/main/src/tfs_frontend/tfs_utils.cpp
        if data_type == np.float32:
            result = 'DT_FLOAT'
        elif data_type == np.int32:
            result = 'DT_INT32'
        elif data_type == np.int64:
            result = 'DT_INT64'
        elif data_type == str:
            result = 'DT_STRING'
        elif data_type == np.uint8:
            result = 'DT_UINT8'
        else:
            raise NotImplementedError()
        return result

    def validate_meta_grpc(self, model, meta):
        """
            Validates model metadata.
            Parameters:
                model (ModelInfo): model class object
                meta (ModelMetadataResponse): model metadata
        """
        json_data = json.loads(MessageToJson(meta))

        assert meta.model_spec.name == model.name, \
            f"Unexpected model name (expected: {model.name}, " \
            f"detected: {meta.model_spec.name})"
        assert meta.model_spec.version.value == model.version
        assert "signature_def" in meta.metadata
        assert meta.metadata['signature_def'].type_url == \
               'type.googleapis.com/tensorflow.serving.SignatureDefMap'
        def validate(test_data, val_shapes, val_types):
            assert len(test_data) == len(val_shapes), \
                f"Unexpected argument list (shapes; expect: {len(val_shapes)}, " \
                f"detect: {len(test_data)})"
            assert len(test_data) == len(val_types), \
                f"Unexpected argument list (shapes; expect: {len(val_types)}, " \
                f"detect: {len(test_data)})"
            for arg_name, arg_data in test_data.items():
                for test, val_dim in zip(arg_data['tensorShape']['dim'], val_shapes[arg_name]):
                    if int(test['size']) != val_dim:
                        raise InvalidMetadataException(
                            f"Unexpected shape (expected: {val_shapes[arg_name]}, " \
                            f"detected: {arg_data['tensorShape']['dim']})")
                val_type = self.cast_type_to_string(val_types[arg_name])
                assert arg_data['dtype'] == val_type, \
                    f"Unexpected type (expected: {val_type}, detected: {arg_data['dtype']}"

        data = json_data['metadata']['signature_def']['signatureDef']['serving_default']
        validate(
            test_data=data['inputs'], val_shapes=model.input_shapes, val_types=model.input_types
        )
        validate(
            test_data=data['outputs'], val_shapes=model.output_shapes, val_types=model.output_types
        )

    def validate_meta_rest(self, model, response):
        metadata = json.loads(response.text)
        assert model.name == metadata['modelSpec']['name']
        assert model.version == int(metadata['modelSpec']['version'])

        metadata_inputs = metadata['metadata']['signature_def']['signatureDef']['serving_default']['inputs']
        metadata_outputs = metadata['metadata']['signature_def']['signatureDef']['serving_default']['outputs']

        for name, description in model.inputs.items():
            assert name in metadata_inputs
            assert model.inputs[name]['shape'] == [
                int(x['size']) for x in metadata_inputs[name]['tensorShape']['dim']
            ]
            assert self.cast_type_to_string(model.inputs[name]['dtype']) == metadata_inputs[name]['dtype']

        for name, description in model.outputs.items():
            assert name in metadata_outputs
            assert model.outputs[name]['shape'] == [
                int(x['size']) for x in metadata_outputs[name]['tensorShape']['dim']
            ]
            assert self.cast_type_to_string(model.outputs[name]['dtype']) == metadata_outputs[name]['dtype']

    def validate_status(self, model, status):
        to_check = status.model_version_status[0]
        assert model.version == to_check.version, f"Unexpected version (detected: {to_check.version}, expected: "\
                                                  f"{model.version})"
        expected_res = get_model_status_pb2.ModelVersionStatus.State.AVAILABLE
        state_map = get_model_status_pb2.ModelVersionStatus.State.items()
        assert to_check.state == expected_res, f"Unexpected state (detected: {to_check.state}, expected "\
                                               f"{expected_res} - map: {state_map})"
        assert to_check.status.error_message == 'OK', f"Unexpected error msg (detected: "\
                                                      f"{to_check.status.error_message}, expected: OK"

