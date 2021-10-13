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

from grpc import RpcError, ssl_channel_credentials, secure_channel, insecure_channel

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest
from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest
from tensorflow_serving.apis.predict_pb2 import PredictRequest

from ovmsclient.tfs_compat.base.serving_client import ServingClient
from ovmsclient.tfs_compat.grpc.requests import (GrpcModelStatusRequest, GrpcModelMetadataRequest,
                                                 GrpcPredictRequest)
from ovmsclient.tfs_compat.grpc.responses import (GrpcModelStatusResponse,
                                                  GrpcModelMetadataResponse,
                                                  GrpcPredictResponse)

from ovmsclient.util.ovmsclient_export import ovmsclient_export


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
            TypeError:  if request argument is of wrong type.
            ValueError: if request argument has invalid contents.
            ConnectionError: if there was an error while sending request to the server.

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

        GrpcClient._check_predict_request(request)

        raw_response = None
        try:
            raw_response = self.prediction_service_stub.Predict(request.raw_request, 10.0)
        except RpcError as e_info:
            raise ConnectionError('There was an error during sending PredictRequest. '
                                  f'Grpc exited with: \n{e_info.code().name} - {e_info.details()}')

        return GrpcPredictResponse(raw_response)

    def get_model_metadata(self, request):
        '''
        Send GrpcModelMetadataRequest to the server and return response..

        Args:
            request: GrpcModelMetadataRequest object.

        Returns:
            GrpcModelMetadataResponse object

        Raises:
            TypeError:  if request argument is of wrong type.
            ValueError: if request argument has invalid contents.
            ConnectionError: if there was an error while sending request to the server.

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

        GrpcClient._check_model_metadata_request(request)

        raw_response = None
        try:
            raw_response = self.prediction_service_stub.GetModelMetadata(request.raw_request, 10.0)
        except RpcError as e_info:
            raise ConnectionError('There was an error during sending ModelMetadataRequest. '
                                  f'Grpc exited with: \n{e_info.code().name} - {e_info.details()}')

        return GrpcModelMetadataResponse(raw_response)

    def get_model_status(self, request):
        '''
        Send GrpcModelStatusRequest to the server and return response.

        Args:
            request: GrpcModelStatusRequest object.

        Returns:
            GrpcModelStatusResponse object

        Raises:
            TypeError:  if request argument is of wrong type.
            ValueError: if request argument has invalid contents.
            ConnectionError: if there was an error while sending request to the server.

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "port": 9000
            ... }
            >>> client = make_grpc_client(config)
            >>> request = make_model_status_request("model")
            >>> response = client.get_model_status(request)
            >>> type(response)
        '''

        GrpcClient._check_model_status_request(request)

        raw_response = None
        try:
            raw_response = self.model_service_stub.GetModelStatus(request.raw_request, 10.0)
        except RpcError as e_info:
            raise ConnectionError('There was an error during sending ModelStatusRequest. '
                                  f'Grpc exited with: \n{e_info.code().name} - {e_info.details()}')

        return GrpcModelStatusResponse(raw_response)

    @classmethod
    def _build(cls, url, tls_config):
        
        ServingClient._check_url(url)

        if tls_config is not None:
            ServingClient._check_tls_config(tls_config)
            server_cert, client_cert, client_key = ServingClient._prepare_certs(
                tls_config.get('server_cert_path'),
                tls_config.get('client_cert_path'),
                tls_config.get('client_key_path')
            )
            creds = ssl_channel_credentials(root_certificates=server_cert,
                                            private_key=client_key, certificate_chain=client_cert)
            channel = secure_channel(url, creds)
        else:
            channel = insecure_channel(url)

        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        model_service_stub = model_service_pb2_grpc.ModelServiceStub(channel)

        return cls(channel, prediction_service_stub, model_service_stub)

    @classmethod
    def _check_model_status_request(cls, request):

        if not isinstance(request, GrpcModelStatusRequest):
            raise TypeError('request type should be GrpcModelStatusRequest, '
                            f'but is {type(request).__name__}')

        if not isinstance(request.raw_request, GetModelStatusRequest):
            raise TypeError('request is not valid GrpcModelStatusRequest')

        if request.raw_request.model_spec.name != request.model_name:
            raise ValueError('request is not valid GrpcModelStatusRequest')

        if request.raw_request.model_spec.version.value != request.model_version:
            raise ValueError('request is not valid GrpcModelStatusRequest')

    @classmethod
    def _check_model_metadata_request(cls, request):

        if not isinstance(request, GrpcModelMetadataRequest):
            raise TypeError('request type should be GrpcModelMetadataRequest, '
                            f'but is {type(request).__name__}')

        if not isinstance(request.raw_request, GetModelMetadataRequest):
            raise TypeError('request is not valid GrpcModelMetadataRequest')

        if request.raw_request.model_spec.name != request.model_name:
            raise ValueError('request is not valid GrpcModelMetadataRequest')

        if request.raw_request.model_spec.version.value != request.model_version:
            raise ValueError('request is not valid GrpcModelMetadataRequest')

        if list(request.raw_request.metadata_field) != ['signature_def']:
            raise ValueError('request is not valid GrpcModelMetadataRequest')

    @classmethod
    def _check_predict_request(cls, request):

        if not isinstance(request, GrpcPredictRequest):
            raise TypeError('request type should be GrpcPredictRequest, '
                            f'but is {type(request).__name__}')

        if not isinstance(request.raw_request, PredictRequest):
            raise TypeError('request is not valid GrpcPredictRequest')

        if request.raw_request.model_spec.name != request.model_name:
            raise ValueError('request is not valid GrpcPredictRequest')

        if request.raw_request.model_spec.version.value != request.model_version:
            raise ValueError('request is not valid GrpcPredictRequest')

        if list(request.inputs.keys()) != list(request.raw_request.inputs.keys()):
            raise ValueError('request is not valid GrpcPredictRequest')


@ovmsclient_export("make_grpc_client", grpcclient="make_client")
def make_grpc_client(url, tls_config=None):
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

    return GrpcClient._build(url, tls_config)
