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

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from requests.auth import AuthBase

from tests.functional.utils.http.http_session import HttpSession


class ClientAuthBase(object, metaclass=ABCMeta):
    """Base class that all http client authentication implementations derive from.

    It performs automatic authentication.
    """
    DATA_OR_BODY = "data"
    _http_auth = None
    request_headers = None
    request_data_params = None
    request_params = None

    def __init__(self, url: str, session: HttpSession, params: dict = None):
        """
        Args:
            url: url
            session: http session
            params: additional params
        """
        self._url = url
        self.session = session
        self.parse_params(params if params is not None else {})
        self._http_auth = self.authenticate()

    @property
    def http_auth(self) -> AuthBase:
        """Http authentication method."""
        return self._http_auth

    @property
    @abstractmethod
    def authenticated(self) -> bool:
        """Is current user already authenticated."""

    @abstractmethod
    def authenticate(self) -> AuthBase:
        """Use session credentials to authenticate."""

    @property
    def request_data(self) -> OrderedDict:
        """Token request data."""
        return OrderedDict([
            ("username", self.session.username),
            ("password", self.session.password),
        ])

    def parse_params(self, params: dict) -> None:
        """params parser to pass configuration customizations"""
        pass

    @property
    def auth_request_params(self) -> dict:
        """build request params"""
        return {
            'url': self._url,
            'headers': self.request_headers,
            self.DATA_OR_BODY: self.request_data,
        }
