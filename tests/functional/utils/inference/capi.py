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

import pytest

from tests.functional.utils.assertions import NotSupported, _assert_status_code_and_message
from tests.functional.utils.inference.communication.base import AbstractCommunicationInterface
from tests.functional.utils.inference.serving.base import AbstractServingWrapper
from tests.functional.data.ovms_capi_wrapper.ovms_capi_shared import OvmsInferenceFailed, OvmsModelNotFound


class CapiServingWrapper(AbstractServingWrapper, AbstractCommunicationInterface):

    NOT_FOUND = OvmsModelNotFound
    INVALID_ARGUMENT = OvmsInferenceFailed
    INTERNAL = None
    ABORTED = None
    RESOURCE_EXHAUSTED = None
    FAILED_PRECONDITION = None
    UNAVAILABLE = None
    ALREADY_EXISTS = None

    def __init__(self, ovms_capi_instance, **kwargs):
        self.ovms_capi_instance = ovms_capi_instance

    def set_grpc_stubs(self):
        raise NotImplementedError()

    def create_inference(self):
        pass    # no special init required

    def predict(self, request, timeout=60, raw=False):
        return self.ovms_capi_instance.send_inference(self.model, request)

    def predict_stream(self):
        raise NotSupported("Streaming API is supported only for KFS:GRPC communication")

    def get_rest_path(self, operation, model_version=None, model_name=None):
        raise NotImplementedError()

    def get_inputs_outputs_from_response(self, response):
        if getattr(self.model, "inputs", None) is None:
            self.model.inputs = {}
        if getattr(self.model, "outputs", None) is None:
            self.model.outputs = {}

        for _input in response['inputs']:
            self.model.inputs[_input['name']] = {
                'shape': _input['shape'],
                'dtype': _input['datatype']
            }

        for output in response['outputs']:
            self.model.outputs[output['name']] = {
                'shape': output['shape'],
                'dtype': output['datatype']
            }

        return

    def get_model_meta_grpc_request(self, model_name=None):
        raise NotImplementedError()

    def get_predict_grpc_request(self):
        raise NotImplementedError()

    def prepare_request(self, input_objects: dict, **kwargs):
        return input_objects

    def get_next_response_from_stream(self, output_stream):
        raise NotSupported("Streaming API is supported only for KFS:GRPC communication")

    def get_model_meta(self, timeout=60, version=None, update_model_info=True, model_name=None):
        response = self.ovms_capi_instance.send_get_model_meta_command(self.model.name, self.model.version)
        if update_model_info:
            self.get_inputs_outputs_from_response(response)
        return response

    def validate_meta(self, model, meta):
        for shape in model.input_shapes.values():
            if not any(shape == _input['shape'] for _input in meta["inputs"]):
                raise Exception(f"Cannot find shape={shape} in meta={meta}")
        for shape in model.output_shapes.values():
            if not any(shape == _output['shape'] for _output in meta["outputs"]):
                raise Exception(f"Cannot find shape={shape} in meta={meta}")

    def get_model_status(self, version=None):
        raise NotSupported("Get model status is not supported in C_API")

    def send_predict_request(self, request, timeout):
        raise NotImplementedError()

    @staticmethod
    def assert_raises_exception(status, error_message_phrase, callable_obj, *args, **kwargs):
        with pytest.raises(status) as e:
            callable_obj(*args, **kwargs)
            _assert_status_code_and_message(status, error_message_phrase,
                                            e.value.status, e.value.error_message, e)
