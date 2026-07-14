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

# pylint: disable=unused-argument
# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-member

import json

from tests.functional.utils.inference.serving.base import AbstractServingWrapper
from tests.functional.utils.logger import get_logger

logger = get_logger(__name__)


class LLMCommonWrapper(AbstractServingWrapper):
    REST_VERSION = "v3"

    def create_base_url(self, rest_version=None):
        rest_version = rest_version if rest_version is not None else self.REST_VERSION
        self.base_url = f"http://{self.url}/{rest_version}"

    def set_grpc_stubs(self):
        raise NotImplementedError

    def create_inference(self):
        self.communication_service = self.create_communication_service()
        return self.communication_service

    def predict(self, request, timeout=300, raw=False):
        result = self.send_predict_request(request, timeout)
        return result

    def get_rest_path(self, operation, model_version=None, model_name=None):
        """
        Expect 1 REST path formats for OpenAI format:
         - POST: (CHAT COMPLETIONS)
            http://{REST_URL}:{REST_PORT}/v3/chat/completions
        """
        rest_path = [self.REST_VERSION, operation]
        rest_path = "/".join(rest_path)
        return rest_path

    def get_inputs_outputs_from_response(self, response):
        pass

    def get_content_from_response(self, response):
        model_specification = json.loads(response.text)
        self.model.content = model_specification["choices"]["message"]["content"]

    def get_model_meta_grpc_request(self, model_name=None):
        # GRPC not supported
        raise NotImplementedError

    def get_predict_grpc_request(self):
        # GRPC not supported
        raise NotImplementedError

    @staticmethod
    def prepare_body_dict_from_request_params(request_params_type, body_dict, **kwargs):
        request_parameters = kwargs.get("request_parameters", None)
        if request_parameters is not None:
            assert isinstance(request_parameters, request_params_type), \
                f"Wrong type of request_parameters expected: {request_params_type} " \
                f"actual: {type(request_parameters)}"
            body_dict.update(request_parameters.prepare_dict(use_extra_body=False))

        return body_dict
