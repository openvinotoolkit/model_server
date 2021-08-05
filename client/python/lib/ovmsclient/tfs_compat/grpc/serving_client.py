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

from grpc import ssl_channel_credentials, secure_channel, insecure_channel
from validators import ipv4, domain
import os
from grpc import RpcError

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest

from ovmsclient.tfs_compat.base.serving_client import ServingClient
from ovmsclient.tfs_compat.grpc.requests import GrpcModelStatusRequest
from ovmsclient.tfs_compat.grpc.responses import GrpcModelStatusResponse

class GrpcClient(ServingClient):

    def __init__(self, channel, prediction_service_stub, model_service_stub):
        self.channel = channel
        self.prediction_service_stub = prediction_service_stub
        self.model_service_stub = model_service_stub

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

        _check_model_status_request(request)

        raw_response = None
        try:
            raw_response = self.model_service_stub.GetModelStatus(request.raw_request, 10.0)
        except RpcError as e_info:
            raise ConnectionError(f'There was an error during sending ModelStatusRequest. Grpc exited with: \n{e_info.code().name} - {e_info.details()}')

        return GrpcModelStatusResponse(raw_response)

    @classmethod
    def _build(cls, config):
        
        _check_config(config)

        grpc_address = f"{config['address']}:{config['port']}"

        if 'tls_config' in config:
            server_cert, client_cert, client_key = _prepare_certs(
                config['tls_config'].get('server_cert_path'),
                config['tls_config'].get('client_cert_path'),
                config['tls_config'].get('client_key_path')
            )
            
            creds = ssl_channel_credentials(root_certificates=server_cert,
            private_key=client_key, certificate_chain=client_cert)

            channel = secure_channel(grpc_address, creds)
        else:
            channel = insecure_channel(grpc_address)

        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        model_service_stub = model_service_pb2_grpc.ModelServiceStub(channel)

        return cls(channel, prediction_service_stub, model_service_stub)

def _check_model_status_request(request):

    if not isinstance(request, GrpcModelStatusRequest):
        raise TypeError(f'request type should be GrpcModelStatusRequest, but is {type(request).__name__}')

    if not isinstance(request.raw_request, GetModelStatusRequest):
        raise TypeError(f'request is not valid GrpcModelStatusRequest')

    if request.raw_request.model_spec.name != request.model_name:
        raise ValueError(f'request is not valid GrpcModelStatusRequest')

    if request.raw_request.model_spec.version.value != request.model_version:
        raise ValueError(f'request is not valid GrpcModelStatusRequest')

def _prepare_certs(server_cert_path, client_cert_path, client_key_path):
    
    client_cert, client_key = None, None

    server_cert = _open_certificate(server_cert_path)

    if client_cert_path is not None:
        client_cert = _open_certificate(client_cert_path)

    if client_key_path is not None:
        client_key = _open_private_key(client_key_path)

    return server_cert, client_cert, client_key

def _open_certificate(certificate_path):
    with open(certificate_path, 'rb') as f:
        certificate = f.read()
        return certificate

def _open_private_key(key_path):
    with open(key_path, 'rb') as f:
        key = f.read()
        return key

def _check_config(config):
    
    if 'address' not in config or 'port' not in config:
        raise ValueError('The minimal config must contain address and port')

    _check_address(config['address'])

    _check_port(config['port'])

    if 'tls_config' in config:
        _check_tls_config(config['tls_config'])
    
def _check_address(address):

    if not isinstance(address, str):
        raise TypeError(f'address type should be string, but is {type(address).__name__}')

    if address != "localhost" and not ipv4(address) and not domain(address):
        raise ValueError(f'address is not valid')

def _check_port(port):
    
    if not isinstance(port, int):
        raise TypeError(f'port type should be int, but is type {type(port).__name__}')

    if port.bit_length() > 16 or port < 0:
        raise ValueError(f'port should be in range <0, {2**16-1}>')

def _check_tls_config(tls_config):

    if 'server_cert_path' not in tls_config:
        raise ValueError(f'server_cert_path is not defined in tls_config')

    if ('client_key_path' in tls_config) != ('client_cert_path' in tls_config):
        raise ValueError(f'none or both client_key_path and client_cert_path are required in tls_config')
    
    valid_keys = ['server_cert_path', 'client_key_path', 'client_cert_path']
    for key in tls_config:
        if not key in valid_keys:
            raise ValueError(f'{key} is not valid tls_config key')
        if not isinstance(tls_config[key], str):
            raise TypeError(f'{key} type should be string but is type {type(tls_config[key]).__name__}')
        if not os.path.isfile(tls_config[key]):
            raise ValueError(f'{tls_config[key]} is not valid path to file')

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
        ValueError, TypeError:  if provided config is invalid.

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
