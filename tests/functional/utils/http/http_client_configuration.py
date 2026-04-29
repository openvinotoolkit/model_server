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

from typing import Union
from urllib.parse import urlparse, urlunparse

from tests.functional.utils.http.base import HttpClientType, HttpUser
from tests.functional.utils.http.exceptions import (
    HttpClientConfigurationEmptyPropertyException,
    HttpClientConfigurationInvalidPropertyTypeException,
)


# pylint: disable=too-many-instance-attributes
class HttpClientConfiguration(object):
    """Http client configuration."""
    identity_attributes = ("client_type", "url", "username", "password")

    # pylint: disable=too-many-arguments
    def __init__(self, client_type: HttpClientType, url: str,
                 username: Union[HttpUser, str] = None, password: str = None,
                 proxies: dict = None, cert: tuple = None, auth_uri: str = None, params: dict = None):
        """
        Args:
            client_type: HttpClientType enum
            url: url to connect to.
            username: user name or HttpUser instance for authentication
            password: user password for authentication
            proxies: proxies
            cert: certs
            auth_uri: authorization uri -> client_auth path or client_auth url for authorization.
                      If path provided auth_url is build from url and path.
            params: additional params if any
        """
        self._validate("client_type", HttpClientType, client_type)
        self._validate("url", str, url)
        self._client_type = client_type
        self._url = url
        self._auth_url = self._prepare_auth_url(url, auth_uri) if auth_uri is not None else None
        self._username = username
        self._password = password
        self.proxies = proxies
        self.cert = cert
        self.params = params

    def _prepare_auth_url(self, url: str, auth_uri: str) -> str:
        """
        Build client_auth url. If auth_uri is url use it, if not build auth_url based on url and auth_uri
        Args:
            url: host url
            auth_uri: authorization path or authorization url if different from host url

        Returns: str authorization url.
        """
        self._auth_uri = urlparse(auth_uri)
        if not self._auth_uri.netloc:
            url = urlparse(url)
            auth_url = urlunparse((url.scheme, url.netloc,
                                   self._auth_uri.path, self._auth_uri.params,
                                   self._auth_uri.query, self._auth_uri.fragment))
        else:
            auth_url = auth_uri
        return auth_url

    def __eq__(self, other):
        return all(getattr(self, a) == getattr(other, a) for a in self.identity_attributes)

    def __hash__(self):
        return hash(tuple(getattr(self, a) for a in self.identity_attributes))

    @property
    def client_type(self):
        """Client type."""
        return self._client_type

    @property
    def url(self):
        """Client api url address."""
        return self._url

    @property
    def auth_url(self):
        """Client authorization path"""
        return self._auth_url

    @property
    def username(self):
        """Client client_auth username."""
        return self._username

    @property
    def password(self):
        """Client client_auth password."""
        return self._password

    @property
    def as_dict(self) -> dict:
        kwargs = dict()
        self.set_value(kwargs, "username", self.username)
        self.set_value(kwargs, "password", self.password)
        self.set_value(kwargs, "proxies", self.proxies)
        self.set_value(kwargs, "cert", self.cert)
        self.set_value(kwargs, "auth_url", self.auth_url)
        self.set_value(kwargs, "params", self.params)
        return kwargs

    @staticmethod
    def set_value(d: dict, key: str, value):
        if value is not None:
            d[key] = value

    @staticmethod
    def _validate(property_name, property_type, property_value):
        """Validate if given property has valid type and value."""
        if not property_value:
            raise HttpClientConfigurationEmptyPropertyException(property_name)
        if not isinstance(property_value, property_type):
            raise HttpClientConfigurationInvalidPropertyTypeException(property_name)