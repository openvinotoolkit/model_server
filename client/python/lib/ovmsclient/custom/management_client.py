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


class ServerManagementClient():

    def reload_servables(self):
        '''
        Send configuration reload request to the server and returns post reload configuration status.
        Requires HTTP interface enabled.

        Returns:
            ConfigStatusResponse object with all models and their versions statuses

        Raises:
            Exceptions for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "port": 9000
            ... }
            >>> client = make_management_client(config)
            >>> response = client.reload_servables()
            >>> type(response)
        '''

        raise NotImplementedError

    def get_servables(self):
        '''
        Send configuration status request to the server.
        Requires HTTP interface enabled.

        Returns:
            ConfigStatusResponse object with all models and their versions statuses

        Raises:
            Exceptions for different serving reponses...

        Examples:

            >>> config = {
            ...     "address": "localhost",
            ...     "port": 9000
            ... }
            >>> client = make_management_client(config)
            >>> response = client.get_servables()
            >>> type(response)
        '''

        raise NotImplementedError

    @classmethod
    def _build(cls, config):
        raise NotImplementedError


def make_management_client(config):
    '''
    Create ServerManagementClient object.

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
        ServerManagementClient object

    Raises:
        ValueError:  if provided config is invalid.

    Examples:
        Create minimal ServerManagementClient:

        >>> config = {
        ...     "address": "localhost",
        ...     "port": 9000
        ... }
        >>> client = make_management_client(config)
        >>> print(client)

        Create ServerManagementClient with TLS:

        >>> config = {
        ...     "address": "localhost",
        ...     "port": 9000,
        ...     "tls_config": {
        ...         "client_key_path": "/opt/tls/client.key",
        ...         "client_cert_path": "/opt/tls/client.crt",
        ...         "server_cert_path": "/opt/tls/server.crt"    
        ...      }
        ... }
        >>> client = make_management_client(config)
        >>> print(client)
    '''
    return ServerManagementClient._build(config)
