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

from tests.functional.utils.http.base import HttpClientType
from tests.functional.utils.http.client_auth.auth import ClientAuthFactory, ClientAuthType
from tests.functional.utils.http.exceptions import HttpClientFactoryInvalidClientTypeException
from tests.functional.utils.http.http_client import HttpClient
from tests.functional.utils.http.http_client_configuration import HttpClientConfiguration


class HttpClientFactory(object):
    """Http client factory with implemented singleton behaviour for each generated client."""

    _INSTANCES = {}

    @classmethod
    def get(cls, configuration: HttpClientConfiguration) -> HttpClient:
        """Create http client for given configuration."""
        client_type = configuration.client_type

        if client_type == HttpClientType.TOKEN_AUTH:
            return cls._get_instance(configuration, ClientAuthType.TOKEN_AUTH)

        elif client_type == HttpClientType.SESSION_AUTH:
            return cls._get_instance(configuration, ClientAuthType.HTTP_SESSION)

        elif client_type == HttpClientType.NO_AUTH:
            return cls._get_instance(configuration, ClientAuthType.NO_AUTH)

        elif client_type == HttpClientType.K8S:
            return cls._get_instance(configuration, ClientAuthType.TOKEN_NO_AUTH)

        elif client_type == HttpClientType.BROKER:
            return cls._get_instance(configuration, ClientAuthType.HTTP_BASIC)

        elif client_type == HttpClientType.BASIC_AUTH:
            return cls._get_instance(configuration, ClientAuthType.HTTP_BASIC)

        elif client_type == HttpClientType.API:
            return cls._get_instance(configuration, ClientAuthType.LOGIN_PAGE)

        elif client_type == HttpClientType.OAUTH2_PROXY_AUTH:
            return cls._get_instance(configuration, ClientAuthType.OAUTH2_PROXY_AUTH)

        elif client_type == HttpClientType.SSL:
            return cls._get_instance(configuration, ClientAuthType.SSL)

        else:
            raise HttpClientFactoryInvalidClientTypeException(client_type)

    @classmethod
    def remove(cls, configuration: HttpClientConfiguration):
        """Remove client instance from cached instances."""
        if configuration in cls._INSTANCES:
            del cls._INSTANCES[configuration]

    @classmethod
    def _get_instance(cls, configuration: HttpClientConfiguration, auth_type):
        """Check if there is already created requested client type and return it otherwise create new instance."""
        if configuration in cls._INSTANCES:
            return cls._INSTANCES[configuration]
        return cls._create_instance(configuration, auth_type)

    @classmethod
    def _create_instance(cls, configuration: HttpClientConfiguration, auth_type):
        """Create new client instance."""
        auth = ClientAuthFactory.get(auth_type=auth_type, **configuration.as_dict)
        instance = HttpClient(configuration.url, auth)
        cls._INSTANCES[configuration] = instance
        return instance
