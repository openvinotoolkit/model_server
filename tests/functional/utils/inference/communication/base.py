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

import abc

from common_libs.ssl import SslCertificates
from tests.functional.object_model.test_environment import TestEnvironment


class AbstractCommunicationInterface(metaclass=abc.ABCMeta):
    def __init__(self, port: int = None, address: str = None,
                 ssl_certificates: SslCertificates = None, **kwargs):
        self.ssl_certificates = ssl_certificates
        self.address = TestEnvironment.get_server_address() if address is None else address
        self.port = port
        self.url = f"{self.address}:{self.port}"

    @abc.abstractmethod
    def prepare_request(self, input_objects: dict, **kwargs):
        """
            Abstract method for preparing the inference request.
        """
        pass

    @abc.abstractmethod
    def get_model_meta(self, timeout=60, version=None, update_model_info=True, model_name=None):
        pass

    @abc.abstractmethod
    def get_model_status(self, model_name=None):
        pass

    @abc.abstractmethod
    def send_predict_request(self, request, timeout):
        pass

    @staticmethod
    @abc.abstractmethod
    def assert_raises_exception(status, error_message_phrase, callable_obj, *args, **kwargs):
        pass
