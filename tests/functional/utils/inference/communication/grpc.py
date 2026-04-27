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

import grpc
from grpc._channel import _InactiveRpcError

from tests.functional.constants.ovms import Ovms

from tests.functional.utils.assertions import AccuracyException, NotSupported, assert_raises_grpc_exception
from tests.functional.utils.inference.communication.base import AbstractCommunicationInterface
from tests.functional.utils.inference.communication.constants import NOT_A_NUMBER_REGEX

GRPC_TIMEOUT = 60
GRPC = "grpc"
channel_options = [('grpc.max_message_length', 100 * 1024 * 1024),
                   ('grpc.max_send_message_length ', 100 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 100 * 1024 * 1024)]


class GrpcErrorCode:
    """
        Defines available gRPC status codes.
    """
    INVALID_ARGUMENT = grpc.StatusCode.INVALID_ARGUMENT.value[0]
    OK = grpc.StatusCode.OK.value[0]
    CANCELLED = grpc.StatusCode.CANCELLED.value[0]
    UNKNOWN = grpc.StatusCode.UNKNOWN.value[0]
    DEADLINE_EXCEEDED = grpc.StatusCode.DEADLINE_EXCEEDED.value[0]
    UNAVAILABLE = grpc.StatusCode.UNAVAILABLE.value[0]
    UNIMPLEMENTED = grpc.StatusCode.UNIMPLEMENTED.value[0]
    ABORTED = grpc.StatusCode.ABORTED.value[0]
    RESOURCE_EXHAUSTED = grpc.StatusCode.RESOURCE_EXHAUSTED.value[0]
    NOT_FOUND = grpc.StatusCode.NOT_FOUND.value[0]
    INTERNAL = grpc.StatusCode.INTERNAL.value[0]
    FAILED_PRECONDITION = grpc.StatusCode.FAILED_PRECONDITION.value[0]
    ALREADY_EXISTS = grpc.StatusCode.ALREADY_EXISTS.value[0]


class GrpcCommunicationInterface(AbstractCommunicationInterface):
    type = GRPC

    NOT_FOUND = GrpcErrorCode.NOT_FOUND
    INVALID_ARGUMENT = GrpcErrorCode.INVALID_ARGUMENT
    INTERNAL = GrpcErrorCode.INTERNAL
    ABORTED = GrpcErrorCode.ABORTED
    RESOURCE_EXHAUSTED = GrpcErrorCode.RESOURCE_EXHAUSTED
    FAILED_PRECONDITION = GrpcErrorCode.FAILED_PRECONDITION
    UNAVAILABLE = GrpcErrorCode.UNAVAILABLE
    ALREADY_EXISTS = GrpcErrorCode.ALREADY_EXISTS
    UNKNOWN = GrpcErrorCode.UNKNOWN

    DEFAULT_EXCEPTION = _InactiveRpcError

    def get_grpc_channel(self):
        """
            Creates gRPC channel with given inference input options.
            Returns:
                channel (Channel)
        """
        if self.ssl_certificates is not None:
            creds = self.ssl_certificates.get_grpc_ssl_channel_credentials()
            channel = grpc.secure_channel(target=self.url, options=channel_options, credentials=creds)
        else:
            channel = grpc.insecure_channel(self.url, options=channel_options)
        return channel

    def create_communication_service(self):
        """
            Method for creating GRPC client.
        """
        self.channel = self.get_grpc_channel()
        self.set_grpc_stubs()
        if self.model_meta_from_serving:
            self.get_model_meta()

    @staticmethod
    def assert_raises_exception(status, error_message_phrase, callable_obj, context=None, *args, **kwargs):
        """
            Check if callable_obj returns specific exception.
            Returns:
                assert_raises_http_exception
        """
        return assert_raises_grpc_exception(
            status, error_message_phrase, callable_obj, context, *args, **kwargs
        )

    def prepare_request(self, input_objects: dict, raw=False, mediapipe_name=None, **kwargs):
        raw = True if self.model.is_mediapipe else raw
        request = self.get_predict_grpc_request(input_objects, raw, mediapipe_name)
        if "context" in kwargs:
            kwargs['context'].request = request
        return {'request': request}

    @classmethod
    def prepare_body_format(cls, input_objects: dict, request_format=Ovms.BINARY_IO_LAYOUT_ROW_NAME):
        """
            Returns request's body dictionary as json data.
        """
        data_obj = cls.prepare_body_dict(input_objects, request_format)
        data_json = json.dumps(data_obj)
        return data_json

    def send_predict_request(self, request: dict, timeout: int, version: int or None=None) -> dict:
        """
        Sends request to server and returns the response as a dictionary.
        :param dict request:
        :param int timeout:
        :param int or None version:
        :rtype: dict
        """
        result = self.send_predict_grpc_request(request['request'], timeout)
        return result

    def process_predict_output(self, result, raw=False):
        r"""
            Example transformation:
                result = {PredictResponse} outputs {\n
                    key: "softmax_tensor"\n
                    value {\n
                        dtype: DT_FLOAT\n    tensor_shape {\n      dim {\n        size: 1\n      }\n
                        dim {\n        size: 1001\n      }\n    }\n
                        tensor_content: "q\25309\233 (...)
                 DESCRIPTOR = {MessageDescriptor} <google.protobuf.pyext._message.MessageDescriptor object
                    at 0x7ff115b38460>
                 OutputsEntry = {GeneratedProtocolMessageType}
                    <class 'tensorflow_serving.apis.predict_pb2.OutputsEntry'>
                 model_spec = {ModelSpec}
                 outputs = {MessageMapContainer: 1} {
                    'softmax_tensor':
                        dtype: DT_FLOAT\n
                        tensor_shape {\n  dim {\n    size: 1\n  }\n  dim  {\n    size: 1001\n  }\n}\n
                        tensor_content: "q\25309\233\336\ (...)

        """
        outputs = self.process_predict_grpc_output(result, raw=raw)
        # check if there are unexpected values in output
        if NOT_A_NUMBER_REGEX.search(str(outputs)):
            raise AccuracyException(f"NaN values found in output: {str(outputs)}")
        return outputs

    def get_model_meta(self, timeout=60, version=None, update_model_info=True, model_name=None):
        """
            Gets information about model metadata.
        """
        request = self.get_model_meta_grpc_request(model_name=model_name)
        response = self.send_model_meta_grpc_request(request)
        if update_model_info:
            self.set_serving_inputs_outputs_grpc(response, model_name=model_name)
        return response

    def get_model_status(self, timeout=60, version=None, model_name=None):
        request = self.get_model_status_grpc_request(model_name=model_name, version=version)
        response = self.send_model_status_grpc_request(request)
        return response

    def get_metrics(self):
        raise NotSupported("Metrics can be loaded only via REST")

    def prepare_stateful_request(self, input_objects: dict, sequence_ctrl=None, sequence_id=None,
                                 ctrl_dtype=None, id_dtype=None):
        return self.prepare_stateful_request_grpc(input_objects, sequence_ctrl, sequence_id, ctrl_dtype, id_dtype)

    def predict_stateful_request(self, request, timeout):
        return self.predict_stateful_request_grpc(request['request'], timeout)

    def is_server_live(self):
        return self.is_server_live_grpc()

    def is_server_ready(self):
        return self.is_server_ready_grpc()

    def is_model_ready(self, model_name, model_version=""):
        return self.is_model_ready_grpc(model_name, model_version)

    def validate_meta(self, model, meta):
        return self.validate_meta_grpc(model, meta)

    def get_server_metadata(self, name=None, version=None):
        return self.get_server_metadata_grpc(name, version)
