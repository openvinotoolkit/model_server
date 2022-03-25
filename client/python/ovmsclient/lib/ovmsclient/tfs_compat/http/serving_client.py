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
from http import HTTPStatus
import requests

from ovmsclient.util.ovmsclient_export import ovmsclient_export
from ovmsclient.tfs_compat.base.serving_client import ServingClient
from ovmsclient.tfs_compat.http.requests import (make_status_request,
                                                 make_metadata_request,
                                                 make_predict_request)
from ovmsclient.tfs_compat.http.responses import (HttpModelStatusResponse,
                                                  HttpModelMetadataResponse,
                                                  HttpPredictResponse)

from ovmsclient.tfs_compat.base.errors import (BadResponseError, ModelServerError,
                                               raise_from_http, raise_from_http_response)


class HttpClient(ServingClient):

    def __init__(self, url, session, client_key=None, server_cert=None):
        self.url = url
        self.session = session
        self.client_key = client_key
        self.server_cert = server_cert

    def predict(self, inputs, model_name, model_version=0, timeout=10.0):
        self._validate_timeout(timeout)
        request = make_predict_request(inputs, model_name, model_version)
        raw_response = None
        try:
            raw_response = self.session.post(f"http://{self.url}"
                                             f"/v1/models/{request.model_name}"
                                             f"/versions/{request.model_version}:predict",
                                             data=request.parsed_inputs,
                                             cert=self.client_key, verify=self.server_cert,
                                             timeout=timeout)
        except requests.exceptions.RequestException as http_error:
            raise_from_http(http_error)

        try:
            response = HttpPredictResponse(raw_response).to_dict()
        except ModelServerError as model_server_error:
            raise model_server_error
        except (JSONDecodeError, ValueError) as parsing_error:
            if raw_response.status_code is not HTTPStatus.OK:
                error_code = HTTPStatus(raw_response.status_code)
                error_message = f"{error_code.value} {error_code.phrase}"
                raise_from_http_response(error_code, error_message)
            raise BadResponseError("Received response is malformed and could not be parsed."
                                   f"Details: {str(parsing_error)}")
        return response["outputs"]

    def get_model_metadata(self, model_name, model_version=0, timeout=10.0):
        self._validate_timeout(timeout)
        request = make_metadata_request(model_name, model_version)
        raw_response = None

        try:
            raw_response = self.session.get(f"http://{self.url}"
                                            f"/v1/models/{request.model_name}"
                                            f"/versions/{request.model_version}"
                                            f"/metadata",
                                            cert=self.client_key, verify=self.server_cert,
                                            timeout=timeout)
        except requests.exceptions.RequestException as http_error:
            raise_from_http(http_error)

        try:
            response = HttpModelMetadataResponse(raw_response).to_dict()
        except ModelServerError as model_server_error:
            raise model_server_error
        except (JSONDecodeError, KeyError, ValueError) as parsing_error:
            if raw_response.status_code is not HTTPStatus.OK:
                error_code = HTTPStatus(raw_response.status_code)
                error_message = f"{error_code.value} {error_code.phrase}"
                raise_from_http_response(error_code, error_message)
            raise BadResponseError("Received response is malformed and could not be parsed."
                                   f"Details: {str(parsing_error)}")
        return response

    def get_model_status(self, model_name, model_version=0, timeout=10.0):
        self._validate_timeout(timeout)
        request = make_status_request(model_name, model_version)
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
            response = HttpModelStatusResponse(raw_response).to_dict()
        except ModelServerError as model_server_error:
            raise model_server_error
        except (JSONDecodeError, KeyError, ValueError) as parsing_error:
            if raw_response.status_code is not HTTPStatus.OK:
                error_code = HTTPStatus(raw_response.status_code)
                error_message = f"{error_code.value} {error_code.phrase}"
                raise_from_http_response(error_code, error_message)
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


@ovmsclient_export("make_http_client", httpclient="make_client")
def make_http_client(url, tls_config=None):
    '''
    Create HttpClient object.

    Args:
        url - Model Server URL as a string in format `<address>:<port>`
        tls_config (optional): dictionary with TLS configuration. The accepted format is:

            .. code-block::

                {
                    "client_key_path": <Path to client key file>,
                    "client_cert_path": <Path to client certificate file>,
                    "server_cert_path": <Path to server certificate file>
                }

            With following types accepted:

            ==================  ==========
            client_key_path     string
            client_cert_path    string
            server_cert_path    string
            ==================  ==========

    Returns:
        HttpClient object

    Raises:
        ValueError, TypeError:  if provided config is invalid.

    Examples:
        Create minimal HttpClient:
        >>> client = make_http_client("localhost:9000")

        Create HttpClient with TLS:

        >>> tls_config = {
        ...     "client_key_path": "/opt/tls/client.key",
        ...     "client_cert_path": "/opt/tls/client.crt",
        ...     "server_cert_path": "/opt/tls/server.crt"
        ... }
        >>> client = make_http_client("localhost:9000", tls_config)
    '''
    return HttpClient._build(url, tls_config)
