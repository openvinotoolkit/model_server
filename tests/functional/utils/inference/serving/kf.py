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

import enum
import json
from http import HTTPStatus

import grpc
import numpy as np
from google.protobuf.json_format import MessageToJson
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype

from tests.functional.utils.assertions import StreamingApiException
from common_libs.http.base import HttpMethod
from tests.functional.utils.inference.communication.grpc import GRPC_TIMEOUT
from tests.functional.utils.inference.serving.base import AbstractServingWrapper
from tests.functional.utils.logger import get_logger
from tests.functional.utils.test_framework import FrameworkMessages, skip_if_runtime
from ovms.constants.metrics import Metric
from tests.functional.constants.ovms import Ovms

logger = get_logger(__name__)

KFS = "KFS"


class DataType(enum.Enum):
    """
        A set of KServe data types bound to auto() values.
    """
    INVALID = enum.auto()
    BOOL = enum.auto()
    UINT8 = enum.auto()
    UINT16 = enum.auto()
    UINT32 = enum.auto()
    UINT64 = enum.auto()
    INT8 = enum.auto()
    INT16 = enum.auto()
    INT32 = enum.auto()
    INT64 = enum.auto()
    FP16 = enum.auto()
    FP32 = enum.auto()
    FP64 = enum.auto()
    STRING = enum.auto()
    BYTES = enum.auto()


class KserveWrapper(AbstractServingWrapper):
    REST_VERSION = "v2"
    PREDICT = "infer"
    MODEL_READY = "ready"

    METRICS_PROTOCOL = Metric.KServe

    def __init__(self, model_meta_from_serving: bool = True, **kwargs):
        super().__init__(model_meta_from_serving=model_meta_from_serving, **kwargs)
        self.model_service_stub = None

    def set_grpc_stubs(self):
        """
            Assigns objects for inference purposes.
        """
        self.model_service_stub = service_pb2_grpc.GRPCInferenceServiceStub(self.channel)

    def create_inference(self):
        """
            Assigns objects for inference purposes.
        """
        self.create_communication_service()

        if self.model_meta_from_serving:
            self.get_model_meta()

    def send_predict_grpc_request(self, request, timeout=GRPC_TIMEOUT):
        return self.model_service_stub.ModelInfer(request)

    @staticmethod
    def process_predict_grpc_output(response, raw=False):
        outputs = {}
        for idx, output in enumerate(response.outputs):
            np_array = np.frombuffer(response.raw_output_contents[idx], dtype=triton_to_np_dtype(output.datatype))
            outputs[output.name] = np_array.reshape(output.shape)
        return outputs

    def predict(self, request, timeout=60, raw=False):
        result = self.send_predict_request(request, timeout)
        outputs = self.process_predict_output(result, raw=raw)
        return outputs

    def predict_stream(self, streaming_generator, inference_requests):
        output_stream = self.model_service_stub.ModelStreamInfer(streaming_generator.main_loop(inference_requests))
        return output_stream

    def get_next_response_from_stream(self, output_stream):
        infer_responses_dict = {}
        for _ in self.outputs:
            try:
                response = next(output_stream, None)
            except grpc._channel._MultiThreadedRendezvous as exc:
                ovms_log = None
                if self.context and self.context.ovms_sessions:
                    ovms = self.context.ovms_sessions[0].ovms
                    ovms_log = ovms.get_logs_as_txt()
                raise StreamingApiException(exc, ovms_log=ovms_log) from exc
            if response is None:
                logger.warning("Attempt of getting response from empty stream.")
            else:
                if not len(response.error_message) == 0:
                    raise StreamingApiException(f"Error message not empty: {response.error_message}")
                infer_responses_dict.update(self.process_predict_output(response.infer_response))
        return infer_responses_dict, response

    def get_model_meta_grpc_request(self, model_name=None):
        version = str(self.model_version) if self.model_version else None
        model_name = model_name if model_name is not None else self.model.name
        return service_pb2.ModelMetadataRequest(name=model_name, version=version)

    def get_predict_grpc_request(self, input_objects, raw=False, mediapipe_name=None):
        request = service_pb2.ModelInferRequest()
        request.model_name = self.model_name if mediapipe_name is None else mediapipe_name
        if self.model_version:
            request.model_version = self.model_version

        request = self.prepare_v2_grpc_input_tensor(request, input_data=input_objects, raw_input_contents=raw)
        return request

    def send_model_meta_grpc_request(self, request):
        metadata = None
        response = self.model_service_stub.ModelMetadata(request=request, metadata=metadata)
        return response

    def get_rest_path(self, operation=None, model_name=None, model_version=None):
        """
            Expect 2 REST path formats for KServe format:
             - GET: (METADATA)
                URL: http://{REST_URL}:{REST_PORT}/v2/models/{model_name}
             - POST: (PREDICT)
                URL: http://{REST_URL}:{REST_PORT}/v2/models/{model_name}/infer

        """
        model_name = model_name if model_name else self.model.name
        assert model_name
        rest_path = [self.REST_VERSION, self.MODELS, model_name]
        if model_version is not None:
            rest_path.append(self.VERSIONS)
            rest_path.append(str(model_version))
        if operation not in [self.METADATA, None]:
            rest_path.append(operation)

        rest_path = "/".join(rest_path)
        return rest_path

    def get_inputs_outputs_from_response(self, response):
        model_specification = json.loads(response.text)

        self.model.inputs = {}
        self.model.outputs = {}

        for _input in model_specification['inputs']:
            self.model.inputs[_input['name']] = {
                'shape': _input['shape'],
                'dtype': triton_to_np_dtype(_input['datatype'])
            }

        for output in model_specification['outputs']:
            self.model.outputs[output['name']] = {
                'shape': output['shape'],
                'dtype': triton_to_np_dtype(output['datatype'])
            }

    @staticmethod
    def prepare_body_dict(input_objects: dict, request_format=Ovms.BINARY_IO_LAYOUT_ROW_NAME, **kwargs):
        """
        example body dict:
                {
                    'request':                  b'{
                                                    "inputs": [{
                                                        "name": "input_name",
                                                        "shape": [],
                                                        "datatype": "BYTES",
                                                        "parameters": {"binary_data_size": "9025"}
                                                    }]
                                                },
                    'inference_header':         {
                                                    'inputs': [{
                                                        'name': 'input_name',
                                                        'shape': [1],
                                                        'datatype': 'BYTES',
                                                        'parameters': {'binary_data_size': '9025'}
                                                    }]
                                                },
                    'inference_header_binary':  b'{
                                                    "inputs": [{
                                                    "name": "input_name",
                                                    "shape": [1],
                                                    "datatype": "BYTES",
                                                    "parameters": {"binary_data_size": "9025"}
                                                    }]
                                                }'
                }
        """
        inputs = []
        for input_name, input_data in input_objects.items():
            if input_data.dtype == np.object_:
                _data = [x.decode() for x in input_data]
            elif input_data.shape:
                _data = input_data.tolist()
            else:
                _data = [input_data.tolist()]

            inputs.append({
                "name": input_name,
                "shape": list(input_data.shape),
                "datatype": "BYTES" if str(input_data.dtype) == "<U4" else np_to_triton_dtype(input_data.dtype),
                "data": _data
            })
        body_dict = {
            'inputs': inputs
        }

        return body_dict

    def process_json_output(self, result_dict):
        """
            Converts predict result to output as a numpy array.
            Input:
                result_dict = {
                 'model_name' = 'resnet-50-tf'
                 'model_version' = '1'
                 'outputs' = [{
                    'name': 'softmax_tensor',
                     'shape': [1, 1001],
                     'datatype': 'FP32',
                     'data': [0.0001684853486949578, ...
                 }]
            Output:
                {'softmax_tensor': {ndarray: (1, 1001)}}
        """
        outputs = {}
        for output in result_dict['outputs']:
            np_array = np.array(output['data'], dtype=triton_to_np_dtype(output['datatype']))
            outputs[output['name']] = np_array.reshape(output['shape'])
        return outputs

    def set_serving_inputs_outputs_grpc(self, response, model_name=None):
        model_name = model_name if model_name is not None else self.model_name
        assert response.name == model_name, f"Cannot find model_name={model_name} in response={response}"
        versions = [int(v) for v in response.versions]

        self.model.inputs = {}
        self.model.outputs = {}
        for resp_input in response.inputs:
            self.model.inputs[resp_input.name] = {
                "shape": [abs(int(i)) for i in resp_input.shape],
                "dtype": triton_to_np_dtype(resp_input.datatype)
            }
            # Aliases for obsolete tests:
            # self.input_names.append(resp_input.name)
            # self.input_dims[resp_input.name] = self.inputs[resp_input.name]["shape"]
            # self.input_data_types[resp_input.name] = self.inputs[resp_input.name]["dtype"]

        for resp_output in response.outputs:
            self.model.outputs[resp_output.name] = {
                "shape": [abs(int(i)) for i in resp_output.shape],
                "dtype": triton_to_np_dtype(resp_output.datatype)
            }
            # self.output_names.append(resp_output.name)
            # self.output_dims[resp_output.name] = self.outputs[resp_output.name]["shape"]
            # self.output_data_types[resp_output.name] = self.outputs[resp_output.name]["dtype"]

        self.metadict = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inputs": self.inputs,
            "outputs": self.outputs
        }

        # if not self.jsonout:
        #     return metadict

        #jout = json.dumps(metadict)
        #print(f"{self.json_prefix}###{self.worker_id}###METADATA###{jout}")
        return self.metadict

    def get_model_status_rest(self, timeout=60, version=None, model_name=None):
        skip_if_runtime(True, FrameworkMessages.KFS_GET_MODEL_STATUS_NOT_SUPPORTED)

    def get_model_status_grpc_request(self, timeout=60, version=None, model_name=None):
        skip_if_runtime(True, FrameworkMessages.KFS_GET_MODEL_STATUS_NOT_SUPPORTED)

    def send_model_status_grpc_request(self, request):
        skip_if_runtime(True, FrameworkMessages.KFS_GET_MODEL_STATUS_NOT_SUPPORTED)

    def is_server_live_grpc(self):
        """
            Gets information about server liveness.

                message ServerLiveRequest {}
                message ServerLiveResponse
                {
                  // True if the inference server is live, false if not live.
                  bool live = 1;
                }
        """
        request = service_pb2.ServerLiveRequest()
        response = self.model_service_stub.ServerLive(request)
        logger.info(f"Server Live: {response.live}")
        return response.live

    def is_server_live_rest(self):
        rest_path = "v2/health/live"
        response = self.client.request(HttpMethod.GET, path=rest_path, timeout=60, raw_response=True)
        return response.status_code == HTTPStatus.OK

    def is_server_ready_grpc(self):
        """
            Gets information about server readiness.

            message ServerReadyRequest {}
            message ServerReadyResponse
            {
              // True if the inference server is ready, false if not ready.
              bool ready = 1;
            }
        """
        request = service_pb2.ServerReadyRequest()
        response = self.model_service_stub.ServerReady(request)
        logger.info(f"Server Ready: {response.ready}")
        return response.ready

    def is_server_ready_rest(self):
        rest_path = "v2/health/ready"
        response = self.client.request(HttpMethod.GET, path=rest_path, timeout=60, raw_response=True)
        return response.status_code == HTTPStatus.OK

    def is_model_ready_grpc(self, model_name, model_version=""):
        request = service_pb2.ModelReadyRequest(name=model_name,
                                                version=str(model_version))
        response = self.model_service_stub.ModelReady(request=request)
        return response.ready

    def is_model_ready_rest(self, model_name, model_version=""):
        """
       Gets information about model readiness (specific only for KFS - gRPC or REST).
       GET http://${REST_URL}:${REST_PORT}/v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready
       Response: True (ready) or False (not ready)
        """
        rest_path = self.get_rest_path(self.MODEL_READY, model_name=model_name, model_version=model_version)
        response = self.client.request(HttpMethod.GET, path=rest_path, timeout=60, raw_response=True)
        return response.status_code == HTTPStatus.OK.value

    def _merge(self, input_list, output=None):
        if output is None:
            if any(type(child) == bytes for child in input_list):
                assert all(
                    type(child) == bytes for child in input_list
                ), "Do not mix types, all inputs should be use bytes"
                output = b''
            else:
                output = []

        for children in input_list:
            if type(children) == list:
                self._merge(children, output)
            elif type(children) == bytes:
                output += children
            else:
                output.append(children)
        return output

    def prepare_v2_grpc_input_tensor(self, request, input_data=None, raw_input_contents=False):
        if input_data is None:
            batch_size = self.model.get_expected_batch_size()
            input_data = self.model.prepare_input_data(batch_size)

        inputs = []

        for input_name in input_data:
            dtype = np_to_triton_dtype(input_data[input_name].dtype)
            shape = input_data[input_name].shape
            input_tensor = service_pb2.ModelInferRequest().InferInputTensor()
            input_tensor.datatype = dtype
            input_tensor.shape.extend(shape)
            input_tensor.name = input_name

            if input_data[input_name].shape:
                merged_input_list = self._merge(input_data[input_name].tolist())
            else:
                merged_input_list = [input_data[input_name].tolist()]

            if raw_input_contents:
                itemsize = input_data[input_name].dtype.itemsize
                data = b''.join(
                    [int(x).to_bytes(length=itemsize, byteorder='little', signed=True) for x in merged_input_list]
                )
                request.raw_input_contents.append(data)
            else:
                self.set_input_v2_grpc_infer_request(input_tensor, merged_input_list, dtype)
            inputs.append(input_tensor)

        request.inputs.extend(inputs)
        return request

    def get_output_contents_v2_grpc_infer_request(self, output):
        """
            Example data:
                output = {
                  ...
                  datatype = "FP32"
                  contents = {
                     ...
                    bool_contents = {RepeatedScalarContainer: 0} []
                    bytes_contents = {RepeatedScalarContainer: 0} []
                    fp32_contents =
                        {RepeatedScalarContainer: 10} [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
                    fp64_contents = {RepeatedScalarContainer: 0} []
                    int64_contents = {RepeatedScalarContainer: 0} []
                    int_contents = {RepeatedScalarContainer: 0} []
                    uint64_contents = {RepeatedScalarContainer: 0} []
                    uint_contents = {RepeatedScalarContainer: 0} []
                  }
                }

            Fetching data example (datatype==FP32):
              content = output.contents.fp32_content[:]
        """

        field_name = self._triton_datatype_to_contents_field_name(output.datatype)
        contents = eval(f"output.contents.{field_name}[:]")
        return contents

    def set_input_v2_grpc_infer_request(self, input_tensor, content, datatype):
        """
            Sets raw input tensor content.
            input_tensor.contents.fp32_contents[:] = content
            Parameters:
                input_tensor (InferInputTensor): inference input tensor
                content (list): inference data content
                datatype (str): data type
            ...
        """
        field_name = self._triton_datatype_to_contents_field_name(datatype)
        input_tensor_var_name = f'{input_tensor=}'.split('=')[0]
        content_var_name = f'{content=}'.split('=')[0]
        if field_name == "bytes_contents":
            expression = f"{input_tensor_var_name}.contents.{field_name}.append({content_var_name})"
        else:
            expression = f"{input_tensor_var_name}.contents.{field_name}[:] = {content_var_name}"
        logger.debug(f"Evaluating expression: '{expression}'")
        # since 'x = y' is a statement, not an expression:
        # use exec (not eval) to run statements.
        exec(expression)

    def _triton_datatype_to_contents_field_name(self, datatype):
        """
            FIELD_NAME:         DATATYPE:
            bool_contents       "BOOL"
            bytes_contents      "BYTES"
            fp32_contents       "FP32"
            fp64_contents       "FP64"
            int64_contents      "INT64"
            int_contents        "INT32"
            uint64_contents     "UINT64"
            uint_contents       "UINT32"
        """
        if datatype == DataType.INT32.name:
            return "int_contents"
        if datatype == DataType.UINT32.name:
            return "uint_contents"
        if datatype == DataType.BYTES.name:
            return "bytes_contents"
        return f"{datatype.lower()}_contents"    # all other types

    def validate_meta_rest(self, model, response):
        metadata = json.loads(response.text)

        assert model.name == metadata['name']
        assert model.version == int(metadata['versions'][0])

        metadata_inputs = metadata['inputs']
        metadata_outputs = metadata['outputs']

        if model.is_mediapipe:
            for input_name in self.model.inputs:
                # Example of meta_input: {'name': 'input', 'datatype': 'INVALID', 'shape': []}
                meta_input = [x for x in metadata_inputs if x['name'] == input_name][0]
                assert meta_input['name'] == input_name
            for output_name in self.model.outputs:
                # Example of meta_output: {'name': 'output', 'datatype': 'INVALID', 'shape': []}
                meta_output = [x for x in metadata_outputs if x['name'] == output_name][0]
                assert meta_output['name'] == output_name
        else:
            for name, description in model.inputs.items():
                meta_input = [x for x in metadata_inputs if x['name'] == name][0]
                assert description['shape'] == meta_input['shape']
                assert self.cast_type_to_string(description['dtype']) == meta_input['datatype']

            for name, description in model.outputs.items():
                meta_output = [x for x in metadata_outputs if x['name'] == name][0]
                assert description['shape'] == meta_output['shape']
                assert description['dtype'] == triton_to_np_dtype(meta_output['datatype'])

    def validate_meta_grpc(self, model, response):
        meta = json.loads(MessageToJson(response))
        metadata_inputs = meta['inputs']
        metadata_outputs = meta['outputs']

        assert model.name == meta['name'], f"Unexpected model name (expected: {model.name}, " \
            f"detected: {meta['name']})"
        assert model.version == int(meta['versions'][0]), f"Unexpected model version (expected: {model.version}, " \
            f"detected: {int(meta['versions'][0])})"

        if model.is_mediapipe:
            for input_name in self.model.inputs:
                meta_input = [x for x in metadata_inputs if x['name'] == input_name][0]
                assert meta_input['name'] == input_name
            for output_name in self.model.outputs:
                meta_output = [x for x in metadata_outputs if x['name'] == output_name][0]
                assert meta_output['name'] == output_name
        else:
            for name, description in model.inputs.items():
                meta_input = [x for x in metadata_inputs if x['name'] == name][0]
                model_shape = [str(i) for i in description['shape']]
                assert model_shape == meta_input['shape'], f"Unexpected model shape (expected: {model_shape}, " \
                f"detected: {meta_input['shape']})"
                assert self.cast_type_to_string(description['dtype']) == meta_input['datatype'], \
                    (f"Unexpected model dtype (expected: {self.cast_type_to_string(description['dtype'])}, "
                     f"detected: {meta_input['datatype']})")

            for name, description in model.outputs.items():
                meta_output = [x for x in metadata_outputs if x['name'] == name][0]
                model_shape = [str(i) for i in description['shape']]
                assert model_shape == meta_output['shape'], f"Unexpected model shape (expected: {model_shape}, " \
                                                            f"detected: {meta_output['shape']})"
                assert description['dtype'] == triton_to_np_dtype(meta_output['datatype']), \
                    (f"Unexpected model dtype (expected: {description['dtype']}, "
                     f"detected: {meta_output['datatype']})")

    def get_server_metadata_grpc(self, name=None, version=None, as_json=True):
        """
            Gets information about server metadata.
            message ServerMetadataRequest {}
            message ServerMetadataResponse
            {
              string name = 1;
              string version = 2;
            }
        """
        request = service_pb2.ServerMetadataRequest()
        response = self.model_service_stub.ServerMetadata(request=request)
        if as_json:
            response = json.loads(MessageToJson(response, preserving_proto_field_name=True))
        if all([name is not None, version is not None]):
            self.validate_kfs_server_meta(response, name, version)
        return response

    def get_server_metadata_rest(self, name=None, version=None, timeout=60):
        """
            Gets information about server metadata.
            Request: GET http://${REST_URL}:${REST_PORT}/v2
            Response:
                If successful:
                    {
                      "name" : $string,
                      "version" : $string,
                    }
                Else:
                    {
                      "error": $string
                    }
        """
        response = self.client.request(
            HttpMethod.GET, path=KserveWrapper.REST_VERSION, timeout=timeout, raw_response=True
        )
        json_response = json.loads(response.text)
        if all([name is not None, version is not None]):
            self.validate_kfs_server_meta(json_response, name, version)
        return json_response

    @staticmethod
    def validate_kfs_server_meta(response, name, version):
        """
            Validates server metadata response (Server Metadata KFS API).
            Parameters:
                response (ServerMetadataResponse)
        """
        assert response['name'] == name
        assert response['version'] == version

    def execute_v2_grpc_infer_request(self, model, input_data=None, raw=False):
        raw = True if model.is_mediapipe else raw
        request = self.get_predict_grpc_request(input_objects=input_data, raw=raw)

        outputs = []
        for out_model_name in model.outputs:
            output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output.name = out_model_name
            outputs.append(output)
        request.outputs.extend(outputs)

        response = self.send_predict_grpc_request(request)

        output_results = []
        for idx, output in enumerate(response.outputs):
            shape = output.shape[:]

            if not response.raw_output_contents or len(response.raw_output_contents) < idx:
                content = self.get_output_contents_v2_grpc_infer_request(output)
            else:
                content = np.frombuffer(response.raw_output_contents[idx], dtype=triton_to_np_dtype(output.datatype))

            np.resize(content, shape)
            output_results.append(content)

        assert len(output_results) == len(model.outputs), f"Expected {len(model.outputs)} output results"

        self.validate_v2_model_name_version(response.model_name, response.model_version, model)
        return response
