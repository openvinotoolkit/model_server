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

import os
from abc import ABC, abstractmethod


class ServingClient(ABC):

    @abstractmethod
    def predict(self, inputs, model_name, model_version=0, timeout=10.0):
        '''
        Send PredictRequest to the server and return response.

        Args:
            inputs: dictionary with (input_name, input data) pairs
            model_name: name of the model in the model server.
            model_version: version of the model (default = 0).
            timeout: time in seconds to wait for the response (default = 10).

        Returns:
            For models with single output - ndarray with prediction result.
            For models with multiple outputs - dictionary of (output_name, result) pairs.

        Raises:
            TypeError:  if provided argument is of wrong type.
            ValueError: if provided argument has unsupported value.
            ConnectionError: if there is an issue with server connection.
            TimeoutError: if request handling duration exceeded timeout.
            ModelNotFound: if model with specified name and version does not exist
                           in the model server.
            InvalidInputError: if provided inputs could not be handled by the model.
            BadResponseError: if server response in malformed and cannot be parsed.
        '''

        pass

    @abstractmethod
    def get_model_metadata(self, model_name, model_version=0, timeout=10.0):
        '''
        Send ModelMetadataRequest to the server and return response.

        Args:
            model_name: name of the model in the model server.
            model_version: version of the model (default = 0).
            timeout: time in seconds to wait for the response (default = 10).

        Returns:
            Dictionary with the model metadata response.

        Raises:
            TypeError:  if provided argument is of wrong type.
            ValueError: if provided argument has unsupported value.
            ConnectionError: if there is an issue with server connection.
            TimeoutError: if request handling duration exceeded timeout.
            ModelNotFound: if model with specified name and version does not exist
                           in the model server.
            BadResponseError: if server response in malformed and cannot be parsed.
        '''

        pass

    @abstractmethod
    def get_model_status(self, model_name, model_version=0, timeout=10.0):
        '''
        Send ModelStatusRequest to the server and return response.

        Args:
            model_name: name of the model in the model server.
            model_version: version of the model (default = 0).
            timeout: time in seconds to wait for the response (default = 10).

        Returns:
            Dictionary with the model status response.

        Raises:
            TypeError:  if provided argument is of wrong type.
            ValueError: if provided argument has unsupported value.
            ConnectionError: if there is an issue with server connection.
            TimeoutError: if request handling duration exceeded timeout.
            ModelNotFound: if model with specified name and version does not exist
                           in the model server.
            BadResponseError: if server response in malformed and cannot be parsed.
        '''

        pass

    @classmethod
    @abstractmethod
    def _build(cls, config):
        raise NotImplementedError

    @classmethod
    def _validate_timeout(cls, timeout):
        try:
            timeout = float(timeout)
            if timeout <= 0.0:
                raise TypeError("timeout set to negative value")
        except Exception:
            raise TypeError("timeout value must be positive float")

    @classmethod
    def _prepare_certs(cls, server_cert_path, client_cert_path, client_key_path):

        client_cert, client_key = None, None

        server_cert = cls._open_certificate(server_cert_path)

        if client_cert_path is not None:
            client_cert = cls._open_certificate(client_cert_path)

        if client_key_path is not None:
            client_key = cls._open_private_key(client_key_path)

        return server_cert, client_cert, client_key

    @classmethod
    def _open_certificate(cls, certificate_path):
        with open(certificate_path, 'rb') as f:
            certificate = f.read()
            return certificate

    @classmethod
    def _open_private_key(cls, key_path):
        with open(key_path, 'rb') as f:
            key = f.read()
            return key

    @classmethod
    def _check_url(cls, url):
        if not isinstance(url, str):
            raise TypeError("url must be a string")

    @classmethod
    def _check_tls_config(cls, tls_config):

        if not isinstance(tls_config, dict):
            raise TypeError('tls_config should be of type dict')

        if 'server_cert_path' not in tls_config:
            raise ValueError('server_cert_path is not defined in tls_config')

        if ('client_key_path' in tls_config) != ('client_cert_path' in tls_config):
            raise ValueError('none or both client_key_path and client_cert_path '
                             'are required in tls_config')

        valid_keys = ['server_cert_path', 'client_key_path', 'client_cert_path']
        for key in tls_config:
            if key not in valid_keys:
                raise ValueError(f'{key} is not valid tls_config key')
            if not isinstance(tls_config[key], str):
                raise TypeError(f'{key} type should be string but is type '
                                f'{type(tls_config[key]).__name__}')
            if not os.path.isfile(tls_config[key]):
                raise ValueError(f'{tls_config[key]} is not valid path to file')
