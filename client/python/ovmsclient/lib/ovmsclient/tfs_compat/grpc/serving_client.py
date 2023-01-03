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

from ovmsclient.tfs_compat.protos.tensorflow_serving.apis import prediction_service_pb2_grpc
from ovmsclient.tfs_compat.protos.tensorflow_serving.apis import model_service_pb2_grpc

from ovmsclient.tfs_compat.base.serving_client import ServingClient
from ovmsclient.tfs_compat.grpc.requests import (make_status_request,
                                                 make_metadata_request,
                                                 make_predict_request)
from ovmsclient.tfs_compat.grpc.responses import (GrpcModelStatusResponse,
                                                  GrpcModelMetadataResponse,
                                                  GrpcPredictResponse)
from ovmsclient.tfs_compat.base.errors import BadResponseError, raise_from_grpc

from ovmsclient.util.ovmsclient_export import ovmsclient_export


class GrpcClient(ServingClient):

    def __init__(self, channel, prediction_service_stub, model_service_stub):
        self.channel = channel
        self.prediction_service_stub = prediction_service_stub
        self.model_service_stub = model_service_stub

    def predict(self, inputs, model_name, model_version=0, timeout=10.0):
        self._validate_timeout(timeout)
        request = make_predict_request(inputs, model_name, model_version)
        raw_response = None

        try:
            raw_response = self.prediction_service_stub.Predict(request.raw_request, timeout)
        except RpcError as grpc_error:
            raise_from_grpc(grpc_error)

        try:
            response = GrpcPredictResponse(raw_response).to_dict()
        except Exception as parsing_error:
            raise BadResponseError("Received response is malformed and could not be parsed."
                                   f"Details: {str(parsing_error)}")
        return response["outputs"]

    def get_model_metadata(self, model_name, model_version=0, timeout=10.0):
        self._validate_timeout(timeout)
        request = make_metadata_request(model_name, model_version)
        raw_response = None
        try:
            raw_response = self.prediction_service_stub.GetModelMetadata(request.raw_request,
                                                                         timeout)
        except RpcError as grpc_error:
            raise_from_grpc(grpc_error)

        try:
            response = GrpcModelMetadataResponse(raw_response).to_dict()
        except Exception as parsing_error:
            raise BadResponseError("Received response is malformed and could not be parsed."
                                   f"Details: {str(parsing_error)}")
        return response

    def get_model_status(self, model_name, model_version=0, timeout=10.0):
        self._validate_timeout(timeout)
        request = make_status_request(model_name, model_version)
        raw_response = None

        try:
            raw_response = self.model_service_stub.GetModelStatus(request.raw_request, timeout)
        except RpcError as grpc_error:
            raise_from_grpc(grpc_error)

        try:
            response = GrpcModelStatusResponse(raw_response).to_dict()
        except Exception as parsing_error:
            raise BadResponseError("Received response is malformed and could not be parsed."
                                   f"Details: {str(parsing_error)}")
        return response

    @classmethod
    def _build(cls, url, tls_config):
        ServingClient._check_url(url)

        # Setting 1 GB max message size.
        MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
        channel_options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]

        if tls_config is not None:
            ServingClient._check_tls_config(tls_config)
            server_cert, client_cert, client_key = ServingClient._prepare_certs(
                tls_config.get('server_cert_path'),
                tls_config.get('client_cert_path'),
                tls_config.get('client_key_path')
            )
            creds = ssl_channel_credentials(root_certificates=server_cert,
                                            private_key=client_key, certificate_chain=client_cert)
            channel = secure_channel(url, creds, channel_options)
        else:
            channel = insecure_channel(url, channel_options)

        prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        model_service_stub = model_service_pb2_grpc.ModelServiceStub(channel)

        return cls(channel, prediction_service_stub, model_service_stub)


@ovmsclient_export("make_grpc_client", grpcclient="make_client")
def make_grpc_client(url, tls_config=None):
    '''
    Create GrpcClient object.

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
        GrpcClient object

    Raises:
        ValueError, TypeError:  if provided config is invalid.

    Examples:
        Create minimal GrpcClient:
        >>> client = make_grpc_client("localhost:9000")

        Create GrpcClient with TLS:

        >>> tls_config = {
        ...     "client_key_path": "/opt/tls/client.key",
        ...     "client_cert_path": "/opt/tls/client.crt",
        ...     "server_cert_path": "/opt/tls/server.crt"
        ... }
        >>> client = make_grpc_client("localhost:9000", tls_config)
    '''

    return GrpcClient._build(url, tls_config)
