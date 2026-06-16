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

import queue

import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http
from tritonclient.utils import np_to_triton_dtype

from tests.functional.utils.assertions import InvalidMetadataException
from tests.functional.utils.inference.communication import GRPC
from tests.functional.utils.inference.communication.base import AbstractCommunicationInterface
from tests.functional.utils.inference.serving.base import AbstractServingWrapper

TRITON = "TRITON"


class TritonServingWrapper(AbstractServingWrapper, AbstractCommunicationInterface):
    def __init__(self, **kwargs):
        self._triton_client = None
        self._requests_sent = 0
        self._async_infer_request_results_queue = queue.Queue()
        AbstractServingWrapper.__init__(self, **kwargs)
        AbstractCommunicationInterface.__init__(self, **kwargs)

    def _async_infer_ready_callback(self, result, error):
        assert error is None, f"Async infer call failed: {error}"
        self._async_infer_request_results_queue.put(result, block=True)

    def get_single_grpc_async_infer_result(self):
        result = self._async_infer_request_results_queue.get(block=True)
        response = result.get_response()
        outputs = {
            output.name: result.as_numpy(output.name) for output in response.outputs
        }
        return outputs

    def get_single_rest_async_infer_result(self, async_request):
        result = async_request.get_result()
        response = result.get_response()
        outputs = {
            output["name"]: result.as_numpy(output["name"]) for output in response['outputs']
        }
        return outputs

    def get_async_infer_results(self, number_of_results):
        if self.type == GRPC:
            result = [self.get_single_grpc_async_infer_result() for i in range(number_of_results)]
        else:
            result = [self.get_single_rest_async_infer_result(async_request) for async_request in self.async_requests]
        return result

    def async_infer(self, request):
        self.async_requests = []
        for _data in request['request']:
            if self.type == GRPC:
                self._triton_client.async_infer(
                    model_name=self.model.name,
                    callback=self._async_infer_ready_callback,
                    inputs=_data)
            else:
                self.async_requests.append(
                    self._triton_client.async_infer(
                        model_name=self.model.name,
                        inputs=_data
                    )
                )
            self._requests_sent += 1

    def infer(self, inputs=None, outputs=None):
        results = self._triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs)
        return results

    def prepare_request(self, input_objects: dict, **kwargs):
        data = []
        for input_name, val in input_objects.items():
            _in = self.api_client.InferInput(input_name, val.shape, np_to_triton_dtype(val.dtype))
            _in.set_data_from_numpy(val)
            data.append(_in)
        return {"request": [data]}

    def get_model_meta(self, timeout=60, version=None, update_model_info=True, model_name=None):
        raise NotImplementedError("Not supported yet")

    def get_model_status(self, model_name=None):
        raise NotImplementedError("Not supported yet")

    def send_predict_request(self, request, timeout):
        raise NotImplementedError("Not supported yet")

    @staticmethod
    def assert_raises_exception(status, error_message_phrase, callable_obj, *args, **kwargs):
        raise NotImplementedError("Not supported yet")

    def set_grpc_stubs(self):
        raise NotImplementedError("Not supported yet")

    def create_inference(self):
        self.api_client = triton_grpc if self.type == GRPC else triton_http
        self._triton_client = self.api_client.InferenceServerClient(url=self.url, verbose=True)

    def predict(self, request):
        raise NotImplementedError("Not supported yet")

    def get_rest_path(self, operation, model_version=None, model_name=None):
        raise NotImplementedError("Not supported yet")

    def get_inputs_outputs_from_response(self, response):
        raise NotImplementedError("Not supported yet")

    def get_model_meta_grpc_request(self, model_name=None):
        raise NotImplementedError("Not supported yet")

    def get_predict_grpc_request(self):
        raise NotImplementedError("Not supported yet")

    def run_triton_infer(self, inputs=None, outputs=None, input_key=None):
        if self.model.is_mediapipe:
            inputs = self.prepare_triton_mediapipe_inputs(input_key) if inputs is None else inputs
            verify_version = False

        else:
            inputs = self.prepare_triton_input_data(inputs) if inputs is None else inputs
            outputs = self.prepare_triton_output_data() if outputs is None else outputs
            verify_version = True

        results = self.infer(inputs=inputs, outputs=outputs)

        response = results.get_response()
        self.validate_triton_response(response, verify_version=verify_version)

    def prepare_triton_input_data(self, input_data=None):
        model_inputs = self.model.inputs if input_data is None else input_data
        inputs = []
        for in_model_name, input_details in model_inputs.items():
            dataset_input = None
            triton_dtype = self.cast_type_to_string(input_details['dtype']) if not self.model.is_language else "BYTES"
            if input_details.get('dataset', None) is not None:
                dataset_input = self.model.prepare_input_data_from_model_datasets()
                shape = input_details['shape'] if not getattr(dataset_input[in_model_name], "shape", None) else \
                dataset_input[in_model_name].shape
            else:
                shape = input_details['shape']

            _input = self.api_client.InferInput(in_model_name, list(shape), triton_dtype)

            if dataset_input is not None:
                input_data = dataset_input[in_model_name]
            else:
                input_data = np.ones(shape=tuple(shape), dtype=model_inputs[in_model_name]['dtype'])
            _input.set_data_from_numpy(input_data)
            inputs.append(_input)

        return inputs

    def prepare_triton_output_data(self):
        outputs = []
        for i, out_model_name in enumerate(self.model.outputs):
            outputs.append(self.api_client.InferRequestedOutput(out_model_name))
        return outputs

    def validate_triton_response(self, response, verify_version=True):
        if self.communication == GRPC:
            response_outputs = response.outputs
            name = response.model_name
            version = response.model_version if verify_version else None
            raw_outputs = response.raw_output_contents
        else:
            response_outputs = response["outputs"]
            name = response['model_name']
            version = response['model_version'] if verify_version else None
            raw_outputs = response_outputs

        error_message = (f"Invalid number of raw_output_contents - Expected: {len(self.model.outputs)}; "
                         f"Actual: {len(raw_outputs)}")
        if not len(response_outputs) == len(self.model.outputs):
            raise InvalidMetadataException(error_message)

        self.validate_v2_model_name_version(name, version, self.model)

    def prepare_triton_mediapipe_inputs(self, input_key=None):
        inputs = []
        input_keys = [input_key] if input_key is not None else self.model.inputs.keys()
        for _input_key in input_keys:
            dataset_input = self.model.prepare_input_data(input_key=_input_key)[_input_key]
            numpy_dataset_input = np.array(dataset_input)

            triton_dtype = self.cast_type_to_string(numpy_dataset_input.dtype)
            shape = numpy_dataset_input.shape
            _input = self.api_client.InferInput(_input_key, list(shape), triton_dtype)
            _input.set_data_from_numpy(numpy_dataset_input)
            inputs.append(_input)

        return inputs
