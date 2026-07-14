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

from enum import Enum

from requests import Session


class HttpMethod(str, Enum):
    """Http request methods."""

    GET = "GET"
    HEAD = "HEAD"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    CONNECT = "CONNECT"
    OPTIONS = "OPTIONS"
    TRACE = "TRACE"
    PATCH = "PATCH"


class HttpClientType(Enum):
    """Http client types."""

    BASIC_AUTH = "BasicAuth"
    SESSION_AUTH = "SessionAuth"
    API = "Api"
    OAUTH2_PROXY_AUTH = "OAuthProxyAuth"
    NO_AUTH = "No Auth"
    BROKER = "Broker"
    K8S = "K8S"
    TOKEN_AUTH = "TokenAuth"
    SSL = "SSL"


class HttpUser:

    def __init__(self, username: str, password: str) -> None:
        self.name = username
        self.password = password
        self._session = None

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id!s}, name={self.name})"

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session: Session):
        self._session = session

    @classmethod
    def from_response(cls, rsp):
        pass

    def delete(self):
        pass

    def is_logged_in(self):
        raise NotImplementedError("Must be implemented in subclass "
                                  "to provide appropriate check "
                                  "if is user logged in before use")
