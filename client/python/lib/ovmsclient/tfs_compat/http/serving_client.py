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

from json.decoder import JSONDecodeError
import requests

from ovmsclient.util.ovmsclient_export import ovmsclient_export
from ovmsclient.tfs_compat.base.serving_client import ServingClient
from ovmsclient.tfs_compat.http.requests import (HttpModelStatusRequest, make_status_request,
                                                 HttpModelMetadataRequest,
                                                 HttpPredictRequest)
from ovmsclient.tfs_compat.http.responses import (HttpModelStatusResponse,
                                                  HttpModelMetadataResponse,
                                                  HttpPredictResponse)

from ovmsclient.tfs_compat.base.errors import BadResponseError, raise_from_http


class HttpClient(ServingClient):

    def __init__(self, url, session, client_key=None, server_cert=None):
        self.url = url
        self.session = session
        self.client_key = client_key
        self.server_cert = server_cert

    def predict(self, request):
        '''
        Send HttpPredictRequest to the server and return response.

        Args:
            request: HttpPredictRequest object.

        Returns:
            HttpPredictResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "port": 5555
            ... }
            >>> client = make_serving_client(config)
            >>> request = make_predict_request({"input": [1, 2, 3]}, "model")
            >>> response = client.predict(request)
            >>> type(response)
        '''

        HttpClient._check_predict_request(request)

        raw_response = None
        try:
            raw_response = self.session.post(f"http://{self.url}"
                                             f"/v1/models/{request.model_name}"
                                             f"/versions/{request.model_version}:predict",
                                             data=request.parsed_inputs,
                                             cert=self.client_key, verify=self.server_cert)
        except requests.exceptions.RequestException as e_info:
            raise ConnectionError('There was an error during sending PredictRequest. '
                                  f'Http exited with:\n{e_info}')

        return HttpPredictResponse(raw_response)

    def get_model_metadata(self, request):
        '''
        Send HttpModelMetadataRequest to the server and return response..

        Args:
            request: HttpModelMetadataRequest object.

        Returns:
            HttpModelMetadataResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "port": 5555
            ... }
            >>> client = make_serving_client(config)
            >>> request = make_model_metadata_request("model")
            >>> response = client.get_model_metadata(request)
            >>> type(response)
        '''

        HttpClient._check_model_metadata_request(request)

        raw_response = None
        try:
            raw_response = self.session.get(f"http://{self.url}"
                                            f"/v1/models/{request.model_name}"
                                            f"/versions/{request.model_version}"
                                            f"/metadata",
                                            cert=self.client_key, verify=self.server_cert)
        except requests.exceptions.RequestException as e_info:
            raise ConnectionError('There was an error during sending ModelMetadataRequest. '
                                  f'Http exited with:\n{e_info}')

        return HttpModelMetadataResponse(raw_response)

    def get_model_status(self, model_name, model_version = 0, timeout = 10.0):

        request = make_status_request(model_name, model_version)

        try:
            timeout = float(timeout)
            if timeout <= 0.0:
                raise
        except:
            raise TypeError("timeout value must be positive float")

        raw_response = None
        try:
            raw_response = self.session.get(f"http://{self.url}"
                                            f"/v1/models/{request.model_name}"
                                            f"/versions/{request.model_version}",
                                            cert=self.client_key, verify=self.server_cert,
                                            timeout=timeout)
        except requests.exceptions.RequestException as http_error:
            raise_from_http(http_error)

        try:
            # to_dict call raises ModelServerError when output JSON contains "error" key
            response = HttpModelStatusResponse(raw_response).to_dict()
        except (JSONDecodeError, KeyError, ValueError) as parsing_error:
            raise BadResponseError("Received response is malformed and could not be parsed."
                                   f"Details: {str(parsing_error)}")
        return response

    @classmethod
    def _build(cls, url, tls_config):
        ServingClient._check_url(url)
        client_cert = None
        server_cert = None
        if tls_config is not None:
            ServingClient._check_tls_config(tls_config)
            if "client_cert_path" in tls_config and "client_key_path" in tls_config:
                client_cert = (tls_config["client_cert_path"], tls_config["client_key_path"])
            server_cert = tls_config.get('server_cert_path', None),
        session = requests.Session()
        return cls(url, session, client_cert, server_cert)

    @classmethod
    def _check_model_status_request(cls, request):
        if not isinstance(request, HttpModelStatusRequest):
            raise TypeError('request type should be HttpModelStatusRequest, '
                            f'but is {type(request).__name__}')

    @classmethod
    def _check_model_metadata_request(cls, request):
        if not isinstance(request, HttpModelMetadataRequest):
            raise TypeError('request type should be HttpModelMetadataRequest, '
                            f'but is {type(request).__name__}')

    @classmethod
    def _check_predict_request(cls, request):
        if not isinstance(request, HttpPredictRequest):
            raise TypeError('request type should be HttpPredictRequest, '
                            f'but is {type(request).__name__}')


@ovmsclient_export("make_http_client", httpclient="make_client")
def make_http_client(url, tls_config=None):
    '''
    Create HttpClient object.

    Args:
        config: Python dictionary with client configuration. The accepted format is:

            .. code-block::

                {
                    "address": <IP address of the serving>,
                    "port": <Port number used by the HTTP interface of the server>,
                        ...more connection options...
                    "tls_config": {
                        "client_key_path": <Path to client key file>,
                        "client_cert_path": <Path to client certificate file>,
                        "server_cert_path": <Path to server certificate file>
                    }
                }

            With following types accepted:

            ==================  ==========
            address             string
            port                integer
            client_key_path     string
            client_cert_path    string
            server_cert_path    string
            ==================  ==========

            The minimal config must contain address and port.

    Returns:
        HttpClient object

    Raises:
        ValueError:  if provided config is invalid.

    Examples:
        Create minimal HttpClient:

        >>> config = {
        ...     "address": "localhost",
        ...     "port": 9000
        ... }
        >>> client = make_http_client(config)
        >>> print(client)

        Create HttpClient with TLS:

        >>> config = {
        ...     "address": "localhost",
        ...     "port": 9000,
        ...     "tls_config": {
        ...         "client_key_path": "/opt/tls/client.key",
        ...         "client_cert_path": "/opt/tls/client.crt",
        ...         "server_cert_path": "/opt/tls/server.crt"
        ...      }
        ... }
        >>> client = make_http_client(config)
        >>> print(client)
    '''
    return HttpClient._build(url, tls_config)
