class ServingClient:
    
    def __init__(self, address, grpc_port, http_port=None):
        raise NotImplementedError

    def send_predict_request(self, request):
        '''
        Send PredictRequest to the server.

        Args:
            request: PredictRequest object. If the request is of type GrpcPredictRequest
                it will be sent over GRPC interface. If the request is of type HttpPredictRequest
                it will be sent over HTTP interface.

        Returns:
            PredictResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "grpc_config": {"port": 9000}
            ... }
            >>> client = make_serving_client(config)
            >>> request = make_predict_request({"input": [1, 2, 3]}, "model")
            >>> response = client.send_predict_request(request)
            >>> type(response)
        '''

        raise NotImplementedError

    def send_model_metadata_request(self, request):
        '''
        Send ModelMetadataRequest to the server.

        Args:
            request: ModelMetadataRequest object. If the request is of type GrpcModelMetadataRequest
                it will be sent over GRPC interface. If the request is of type HttpModelMetadataRequest
                it will be sent over HTTP interface.

        Returns:
            ModelMetadataResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "grpc_config": {"port": 9000}
            ... }
            >>> client = make_serving_client(config)
            >>> request = make_model_metadata_request("model")
            >>> response = client.send_model_metadata_request(request)
            >>> type(response)
        '''

        raise NotImplementedError

    def send_model_status_request(self, request):
        '''
        Send ModelStatusRequest to the server.

        Args:
            request: ModelStatusRequest object. If the request is of type GrpcModelStatusRequest
                it will be sent over GRPC interface. If the request is of type HttpModelStatusRequest
                it will be sent over HTTP interface.

        Returns:
            ModelStatusResponse object

        Raises:
            TypeError:  if provided argument is of wrong type.
            Many more for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "grpc_config": {"port": 9000}
            ... }
            >>> client = make_serving_client(config)
            >>> request = make_model_status_request("model")
            >>> response = client.send_model_status_request(request)
            >>> type(response)
        '''

        raise NotImplementedError

    def send_config_reload_request(self):
        '''
        Send configuration reload request to the server and returns post reload configuration status.
        Requires HTTP interface enabled.

        Returns:
            HttpConfigStatusResponse object with all models and their versions statuses

        Raises:
            Exceptions for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "grpc_config": {"port": 9000}
            ... }
            >>> client = make_serving_client(config)
            >>> response = client.send_config_reload_request()
            >>> type(response)
        '''

        raise NotImplementedError

    def send_config_status_request(self):
        '''
        Send configuration status request to the server.
        Requires HTTP interface enabled.

        Returns:
            HttpConfigStatusResponse object with all models and their versions statuses

        Raises:
            Exceptions for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "grpc_config": {"port": 9000}
            ... }
            >>> client = make_serving_client(config)
            >>> response = client.send_config_status_request()
            >>> type(response)
        '''

        raise NotImplementedError

def make_serving_client(config):
    '''
    Create ServingClient object.

    Args:
        config: Python dictionary with client configuration. The accepted format is:

            .. code-block::

                {
                    "address": <IP address of the serving>,
                    "grpc_config: {
                        "port": <Port number used by the gRPC interface of the server>,
                        ...more connection options...
                    },
                    "http_config: {
                        "port": <Port number used by the HTTP interface of the server>,
                        ...more connection options...
                    },
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
            
            The minimal config must contain address and port for either gRPC or HTTP.

    Returns:
        ServingClient object

    Raises:
        ValueError:  if provided config is invalid.

    Examples:
        Create minimal ServingClient:

        >>> config = {
        ...     "address": "localhost",
        ...     "grpc_config": {"port": 9000}
        ... }
        >>> client = make_serving_client(config)
        >>> print(client)

        Create ServingClient for both GRPC and HTTP with TLS:

        >>> config = {
        ...     "address": "localhost",
        ...     "grpc_config": {"port": 9000},
        ...     "http_config": {"port": 5000},
        ...     "tls_config": {
        ...         "client_key_path": "/opt/tls/client.key",
        ...         "client_cert_path": "/opt/tls/client.crt",
        ...         "server_cert_path": "/opt/tls/server.crt"    
        ...      }
        ... }
        >>> client = make_serving_client(config)
        >>> print(client)
    '''

    raise NotImplementedError

