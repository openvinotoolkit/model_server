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

from ovmsclient.tfs_compat.base.serving_client import ServingClient

class GrpcClient(ServingClient):

    def predict(self, request):
        '''
        Send GrpcPredictRequest to the server and return response.

        Args:
            request: GrpcPredictRequest object.

        Returns:
            GrpcPredictResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "port": 9000
            ... }
            >>> client = make_grpc_client(config)
            >>> request = make_predict_request({"input": [1, 2, 3]}, "model")
            >>> response = client.predict(request)
            >>> type(response)
        '''

        raise NotImplementedError

    def get_model_metadata(self, request):
        '''
        Send GrpcModelMetadataRequest to the server and return response..

        Args:
            request: GrpcModelMetadataRequest object.

        Returns:
            GrpcModelMetadataResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "port": 9000
            ... }
            >>> client = make_grpc_client(config)
            >>> request = make_model_metadata_request("model")
            >>> response = client.get_model_metadata(request)
            >>> type(response)
        '''

        raise NotImplementedError

    def get_model_status(self, request):
        '''
        Send GrpcModelStatusRequest to the server and return response.

        Args:
            request: GrpcModelStatusRequest object.

        Returns:
            GrpcModelStatusResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     :port": 9000
            ... }
            >>> client = make_grpc_client(config)
            >>> request = make_model_status_request("model")
            >>> response = client.get_model_status(request)
            >>> type(response)
        '''

        raise NotImplementedError

    @classmethod
    def _build(cls, config):
        raise NotImplementedError

def make_grpc_client(config):
    '''
    Create GrpcClient object.

    Args:
        config: Python dictionary with client configuration. The accepted format is:

            .. code-block::

                {
                    "address": <IP address of the serving>,
                    "port": <Port number used by the gRPC interface of the server>,
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
        GrpcClient object

    Raises:
        ValueError:  if provided config is invalid.

    Examples:
        Create minimal GrpcClient:

        >>> config = {
        ...     "address": "localhost",
        ...     "port": 9000
        ... }
        >>> client = make_grpc_client(config)
        >>> print(client)

        Create GrpcClient with TLS:

        >>> config = {
        ...     "address": "localhost",
        ...     "port": 9000,
        ...     "tls_config": {
        ...         "client_key_path": "/opt/tls/client.key",
        ...         "client_cert_path": "/opt/tls/client.crt",
        ...         "server_cert_path": "/opt/tls/server.crt"    
        ...      }
        ... }
        >>> client = make_grpc_client(config)
        >>> print(client)
    '''
    return GrpcClient._build(config)
