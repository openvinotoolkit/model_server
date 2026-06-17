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
from http import HTTPStatus

from tests.functional.constants.ovms import Ovms
from tests.functional.utils.assertions import AccuracyException, UnexpectedResponseError, assert_raises_http_exception
from tests.functional.utils.http.base import HttpMethod
from tests.functional.utils.http.client_auth.auth import NoAuthConfigurationProvider, SslAuthConfigurationProvider
from tests.functional.utils.http.http_client_factory import HttpClientFactory
from tests.functional.utils.logger import get_logger
from tests.functional.utils.inference.communication.base import AbstractCommunicationInterface
from tests.functional.utils.inference.communication.constants import NOT_A_NUMBER_REGEX

logger = get_logger(__name__)

REST = 'rest'
HTTP_PROTOCOL = "http://"
HTTPS_PROTOCOL = "https://"


class RestCommunicationInterface(AbstractCommunicationInterface):
    type = REST

    NOT_FOUND = HTTPStatus.NOT_FOUND
    INVALID_ARGUMENT = HTTPStatus.BAD_REQUEST
    INTERNAL = HTTPStatus.INTERNAL_SERVER_ERROR
    FAILED_PRECONDITION = HTTPStatus.PRECONDITION_FAILED
    UNAVAILABLE = HTTPStatus.SERVICE_UNAVAILABLE
    ALREADY_EXISTS = HTTPStatus.CONFLICT
    RESOURCE_EXHAUSTED = HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    DEFAULT_EXCEPTION = UnexpectedResponseError

    METADATA = "metadata"
    MODELS = "models"
    STATUS = "status"
    VERSIONS = "versions"
    METRICS = "metrics"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)  # init generic params

    def create_communication_service(self):
        """
            Method for creating HTTP client.
        """
        if self.ssl_certificates is None:
            assert HTTPS_PROTOCOL not in self.url, \
                "Using https protocol without ssl certificates not allowed. " \
                f"Provided url invalid: {self.url}"
            url = HTTP_PROTOCOL + self.url if HTTP_PROTOCOL not in self.url else self.url
            client = HttpClientFactory.get(NoAuthConfigurationProvider.get(url=url, proxies={}))
        else:
            assert HTTP_PROTOCOL not in self.url, \
                "Using http protocol with ssl certificates not allowed. " \
                f"Provided url invalid: {self.url}"
            url = HTTPS_PROTOCOL + self.url if HTTPS_PROTOCOL not in self.url else self.url
            cert = self.ssl_certificates.get_https_cert()
            client = HttpClientFactory.get(
                SslAuthConfigurationProvider.get(url=url, cert=cert, proxies={}))
        self.client = client
        if self.model_meta_from_serving:
            self.get_model_meta()

    def prepare_request(self, input_objects: dict, request_format=Ovms.BINARY_IO_LAYOUT_ROW_NAME, **kwargs):
        data_json = self.prepare_body_format(input_objects=input_objects, request_format=request_format,
                                             model=self.model)
        if "context" in kwargs:
            kwargs['context'].request = data_json
        return {'request': data_json}

    @classmethod
    def prepare_body_format(cls, input_objects: dict, request_format=Ovms.BINARY_IO_LAYOUT_ROW_NAME, **kwargs):
        """
            Returns request's body dictionary as json data.
        """
        data_obj = cls.prepare_body_dict(input_objects, request_format=request_format, **kwargs)
        data_json = json.dumps(data_obj)
        return data_json

    def send_predict_request(self, request, timeout, version=None):
        version = self.model.version if not version else version
        rest_path = self.get_rest_path(self.PREDICT, model_version=version)

        data = request if type(request) == str else request.get('request', None)
        try:
            headers = request.get('inference_header', None)
        except AttributeError:
            headers = None
        result = self.client.request(HttpMethod.POST,
                                     path=rest_path,
                                     data=data,
                                     headers=headers,
                                     timeout=timeout,
                                     raw_response=True)
        return result

    def get_model_meta(self, timeout=60, version=None, update_model_info=True, model_name=None):
        rest_path = self.get_rest_path(self.METADATA, model_version=version)
        self.model_meta_response = self.client.request(
            HttpMethod.GET, path=rest_path, timeout=timeout, raw_response=True
        )

        if update_model_info:
            self.get_inputs_outputs_from_response(self.model_meta_response)
        return self.model_meta_response

    def get_metrics(self, raw_response=True):
        """
            Gets metrics and returns output as string.
        """
        result = self.client.request(HttpMethod.GET, self.METRICS, raw_response=True)
        return result.text

    def get_model_status(self, timeout=60, version=None, model_name=None):
        if version is None:
            version = self.model.version
        response = self.get_model_status_rest(timeout, version, model_name)
        return response

    def set_serving_inputs_outputs(self, response):
        """
            Sets inference response inputs and outputs.
            Parameters:
                response (GetModelMetadataResponse): inference response
        """
        signature_def = response.metadata['signature_def']
        # signature_map = get_model_metadata_pb2.SignatureDefMap()
        # signature_map.ParseFromString(signature_def.value)
        # serving_default = signature_map.ListFields()[0][1]['serving_default']
        # serving_inputs = serving_default.inputs
        # serving_outputs = serving_default.outputs
        #
        # if self.input_names is None:
        #     self.input_names = list(serving_inputs.keys())
        # if self.output_names is None:
        #     self.output_names = list(serving_outputs.keys())
        #
        # self.input_dims = {}
        # self.input_data_types = {}
        # self.output_dims = {}
        # self.output_data_types = {}
        #
        # for input_name in self.input_names:
        #     serving_input = serving_inputs[input_name]
        #     input_tensor_shape = serving_input.tensor_shape
        #     self.input_dims[input_name] = [d.size for d in input_tensor_shape.dim]
        #     self.input_data_types[input_name] = self.get_data_type(serving_input.dtype)
        #
        # for output_name in self.output_names:
        #     serving_output = serving_outputs[output_name]
        #     output_tensor_shape = serving_output.tensor_shape
        #     self.output_dims[output_name] = [d.size for d in output_tensor_shape.dim]
        #     self.output_data_types[output_name] = self.get_data_type(serving_output.dtype)

    def process_predict_output(self, result, **kwargs):
        # check if there are unexpected values in output (TFS - NaN/ KFS - empty values)
        if NOT_A_NUMBER_REGEX.search(result.text):
            raise AccuracyException(f"NaN values found in output: {result.text}")
        output_json = json.loads(result.text)
        outputs = self.process_json_output(output_json)
        return outputs

    @staticmethod
    def assert_raises_exception(status, error_message_phrase, callable_obj, context=None, *args, **kwargs):
        """
            Check if callable_obj returns specific exception.
            Returns:
                assert_raises_http_exception
        """
        return assert_raises_http_exception(
            status, error_message_phrase, callable_obj, context, *args, **kwargs
        )

    def prepare_stateful_request(self, input_objects: dict, sequence_ctrl=None, sequence_id=None,
                                 ctrl_dtype=None, id_dtype=None):
        return self.prepare_stateful_request_rest(input_objects, sequence_ctrl, sequence_id, ctrl_dtype, id_dtype)

    def predict_stateful_request(self, request, timeout):
        return self.predict_stateful_request_rest(request, timeout)

    def is_server_live(self):
        return self.is_server_live_rest()

    def is_server_ready(self):
        return self.is_server_ready_rest()

    def is_model_ready(self, model_name, model_version=""):
        return self.is_model_ready_rest(model_name, model_version)

    def validate_meta(self, model, meta):
        return self.validate_meta_rest(model, meta)

    def get_server_metadata(self, name=None, version=None):
        return self.get_server_metadata_rest(name, version)
