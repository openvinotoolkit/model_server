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

import pprint
import time
from enum import Enum
from typing import Callable, Union
from urllib.parse import urlparse

import requests
from requests import Response, Session
from requests.auth import AuthBase, HTTPBasicAuth, extract_cookies_to_jar

from tests.functional.utils.assertions import UnexpectedResponseError
from tests.functional.utils.http.base import HttpClientType, HttpMethod, HttpUser
from tests.functional.utils.http.http_client_configuration import HttpClientConfiguration
from tests.functional.utils.http.http_session import HttpSession
from tests.functional.utils.logger import get_logger
from tests.functional.config import http_proxy, https_proxy

from tests.functional.utils.http.client_auth.base import ClientAuthBase
from tests.functional.utils.http.client_auth.exceptions import (
    ClientAuthFactoryInvalidAuthTypeException,
    ClientAuthSessionMissingResponseSessionHeaderException,
)

logger = get_logger(__name__)


class HTTPSessionAuth(AuthBase):
    """Attaches session authentication to the given request object."""

    def __init__(self, session):
        self._session = session

    def __call__(self, request):
        request.headers['Session'] = self._session
        return request


class HTTPTokenAuth(AuthBase):
    """Attaches token authentication to the given request object."""

    def __init__(self, token):
        self._token = token

    def __call__(self, request):
        request.headers['Authorization'] = self._token
        return request


class ClientAuthType(Enum):
    """Client authentication types."""

    HTTP_SESSION = "HttpSession"
    HTTP_BASIC = "HttpBasic"
    NO_AUTH = "NoAuth"
    TOKEN_AUTH = "TokenAuth"
    TOKEN_NO_AUTH = "TokenNoAuth"
    SSL = "SSL"
    LOGIN_PAGE = "Login Page"
    OAUTH2_PROXY_AUTH = "OAuth2ProxyAuth"


class NoAuthConfigurationProvider(object):
    """Provide configuration for no client_auth http client."""

    @classmethod
    def get(cls, url: str, proxies=None) -> HttpClientConfiguration:
        """Provide http client configuration."""
        return HttpClientConfiguration(
            client_type=HttpClientType.NO_AUTH,
            url=url,
            proxies=proxies
        )


class SslAuthConfigurationProvider(object):
    """Provide configuration for https client with SSL/TLS."""

    @classmethod
    def get(cls, url: str, cert: tuple, proxies=None) -> HttpClientConfiguration:
        """Provide http client configuration."""
        return HttpClientConfiguration(
            client_type=HttpClientType.SSL,
            url=url,
            cert=cert,
            proxies=proxies
        )


# pylint: disable=too-many-instance-attributes
class ClientAuthToken(ClientAuthBase):
    """Base class that all token based http client authentication implementations derive from."""
    request_headers = {"Accept": "application/json"}
    token_life_time = None
    token_name = None
    token_header_format = "Bearer {}"

    def __init__(self, url: str, session: HttpSession, params: dict = None):
        self._token = None
        self._token_header = None
        self._token_timestamp = None
        self._response = None
        super().__init__(url, session, params)

    @property
    def token(self) -> str:
        """token is the token retrieved from the response during authentication."""
        return self._token

    @property
    def authenticated(self) -> bool:
        """Check if current user is authenticated."""
        return self._token and not self._is_token_expired()

    def authenticate(self) -> AuthBase:
        """Use session credentials to authenticate."""
        self._response = self.session.request(**self.auth_request_params)
        self._set_token()
        self._http_auth = HTTPTokenAuth(self._token_header)
        return self._http_auth

    def _is_token_expired(self):
        """Check if token has been expired."""
        return time.time() - self._token_timestamp > self.token_life_time

    def _set_token(self):
        """Set token taken from token request response."""
        if self.token_name not in self._response:
            raise ClientAuthTokenMissingResponseTokenKeyException()
        self._token_timestamp = time.time()
        self._token = self._response[self.token_name]
        self._token_header = self.token_header_format.format(self._token)

    def parse_params(self, params: dict):
        self.token_name = params.get("token_name", "access_token")
        self.token_life_time = params.get("token_life_time", 298)
        self.request_data_params = params.get("request_data", {})
        self.request_params = params.get("request_params", {})

    @property
    def auth_request_params(self) -> dict:
        """build request params"""
        request_params = super().auth_request_params
        request_params.update(self.request_params)
        return request_params

    @property
    def request_data(self) -> dict:
        """Token request data."""
        request_data = super().request_data
        request_data.update(self.request_data_params)
        return request_data


class ClientAuthTokenMissingResponseTokenKeyException(Exception):
    """Exception that is thrown when no token is found in the response"""
    def __init__(self):
        super().__init__("Token key is missing in token request response.")



class ClientAuthSession(ClientAuthBase):
    """Base class that all session based http client authentication implementations derive from."""
    DATA_OR_BODY = "body"
    request_headers = {"Content-Type": "application/json"}
    session_life_time = 600

    def __init__(self, url: str, session: HttpSession, params: dict):
        self._session_id = None
        self._session_timestamp = None
        self._response = None
        super().__init__(url, session, params)

    @property
    def session_id(self) -> str:
        """session is the session retrieved from the response during authentication."""
        return self._session_id

    @property
    def authenticated(self) -> bool:
        """Check if current user is authenticated."""
        return self._session_id and not self._is_session_expired()

    def authenticate(self) -> AuthBase:
        """Use session credentials to authenticate."""
        self._response = self.session.request(**self.auth_request_params)
        self._set_session_id()
        self._http_auth = HTTPSessionAuth(self._session_id)
        return self._http_auth

    def _is_session_expired(self):
        """Check if token has been expired."""
        return time.time() - self._session_timestamp > self.session_life_time

    def _set_session_id(self):
        """Set token taken from token request response."""
        self._session_id = self._response.headers.get("Session", None)
        if self._session_id is None:
            raise ClientAuthSessionMissingResponseSessionHeaderException()
        self._session_timestamp = time.time()

    @property
    def auth_request_params(self) -> dict:
        """build request params"""
        request_params = super().auth_request_params
        request_params.update({
            'method': HttpMethod.POST,
            'raw_response': True,
            'log_message': "Retrieve session id.",
        })
        return request_params

    def parse_params(self, params: dict):
        self.session_life_time = params.get("session_life_time", 60 * 10)
        self.request_data_params = params.get("request_data", {})
        self.request_params = params.get("request_params", {})


class ClientAuthHttpBasic(ClientAuthBase):
    """Http basic based http client authentication."""

    def authenticate(self) -> AuthBase:
        """Use session credentials to authenticate."""
        self._http_auth = HTTPBasicAuth(*self.request_data.values())
        return self._http_auth

    @property
    def authenticated(self) -> bool:
        """Check if current user is authenticated."""
        return True



class ClientAuthTokenProvided(ClientAuthBase):
    """Token based http client authentication."""

    def __init__(self, url: str, session: HttpSession, params: dict = None):
        self._token = params['token']
        super().__init__(url, session, params)

    def authenticate(self) -> AuthBase:
        """Use session credentials to authenticate."""
        self._http_auth = HTTPTokenAuth(self._token)
        return self._http_auth

    @property
    def authenticated(self) -> bool:
        """Check if current user is authenticated."""
        return True


class ClientAuthNoAuth(ClientAuthBase):
    """No authentication."""

    def authenticate(self) -> AuthBase:
        pass

    @property
    def authenticated(self) -> bool:
        """always authenticated"""
        return True

class OAuth2ProxyAuth(AuthBase):
    """Fake authorization, installed hooks will handle authorization"""
    def __call__(self, r):
        return r

class ClientAuthSsl(ClientAuthNoAuth):
    """
    Class implemented to comply with coding standard.
    Authorisation is taken care by SSL certificates, no extra auth is needed.
    """
    pass


class HTTPCookieAuth(AuthBase):
    """Attaches session authentication to the given request object."""

    def __init__(self, cookie):
        self._cookie = cookie

    def __call__(self, request):
        for key, value in self._cookie.items():
            if request.headers.get('Cookie', None) is not None:
                request.headers['Cookie'] = f"{request.headers['Cookie']};{key}={value}"
            else:
                request.headers['Cookie'] = f"{key}={value}"

        return request

    @property
    def cookie(self):
        return self._cookie



class ClientAuthLoginPage(ClientAuthBase):
    """Login page based http client authentication."""
    CSRF_NAME = "_oauth2_proxy_csrf"
    PROXY_NAME = "_oauth2_proxy"
    proxies = {
        "http": http_proxy ,
        "https": https_proxy,
        "no_proxy": ""
    } if http_proxy != "" else None

    def authenticate(self) -> AuthBase:
        """Use session credentials to authenticate."""

        response = requests.post(url=self._url,
                                 data=self._request_data(),
                                 cookies=self._request_cookies(),
                                 verify=False,
                                 allow_redirects=True,
                                 proxies=self.proxies)
        if not response.ok:
            raise UnexpectedResponseError(response.status_code, response.text)
        previous_response = response.history[-1]
        cookie = {self.PROXY_NAME: previous_response.cookies.get(self.PROXY_NAME)}

        return HTTPCookieAuth(cookie=cookie)

    @property
    def authenticated(self) -> bool:
        return True

    def parse_params(self, params: dict) -> None:
        """params parser to pass configuration customizations"""
        self.request_params = params

    def _request_headers(self):
        """Prepare request data."""
        csrf_name = "_oauth2_proxy_csrf"
        return {
            "Cookie": f"{csrf_name}={self.request_params[csrf_name]}",
        }

    def _request_cookies(self):
        """Prepare request data."""
        return {self.CSRF_NAME: self.request_params[self.CSRF_NAME]}

    def _request_data(self):
        """Prepare request data."""
        data = {
            "login": self.session.username,
            "password": self.session.password,
        }
        return data


class ClientAuthOAuth2Proxy(ClientAuthBase):
    """Login page based http client authentication."""
    CSRF_NAME = "_oauth2_proxy_csrf"
    PROXY_NAME = "_oauth2_proxy"

    def __init__(self, url: str, session: HttpSession, params: dict = None):
        self.hooks_added = False
        super().__init__(url, session, params)

    def cookies_repr(self, indent: int):
        cookies = []
        for name, value in self.session.cookies.iteritems():
            name_ = f"{name}:"
            space = 24 - indent
            cookies.append(f"{' '*indent}{name_:{space}} {value[:30]}")
        return "\n".join(cookies) + "\n"

    def login_hook(self, http_session: HttpSession) -> Callable[[Response], Response]:
        state = dict(logging_hook_fn=0, redirects=0,
                     authorizations=0, authorizations_skipped=0)

        def login_hook_fn(initial_response: Response, *args, **kwargs) -> Response:
            logger.verbose(f"\nLogin hook for session: {str(id(http_session))[-4:]}."
                           f"User: {http_session.username} "
                           f" state:\n{pprint.pformat(state)}\n"
                           f" proxy:\n{pprint.pformat(http_session.session.proxies)}")
            state["logging_hook_fn"] += 1
            session = http_session.session
            location = initial_response.headers.get("Location")
            if initial_response.status_code == 302 and location is not None:
                state["redirects"] += 1
                next_url = urlparse(location)
                initial_request = initial_response.request
                started_authorization_path = getattr(session, "__authorization_started",
                                                     next_url.path)
                if any(path == next_url.path for path in ["/oauth2/start", "/dex/auth"]) and\
                        started_authorization_path == next_url.path:
                    tt_resend_counter = int(
                        initial_request.headers.get("tt_resend_counter", "1")
                    )
                    if tt_resend_counter < 4 and http_session.user.is_logged_in():
                        state["authorizations_skipped"] += 1
                        logger.verbose(
                            f"\nLogin hook for session: {str(id(http_session))[-4:]}.\n"
                            f"User: {http_session.username} "
                            f"re-send counter is {tt_resend_counter}, re-using cookies.")
                        return re_send_initial_request(initial_response, session)
                    logger.verbose(f"re-send counter is {tt_resend_counter}, re-authorizing.")
                    state["authorizations"] += 1
                    logger.verbose(
                        f"\nLogin hook for session: {str(id(http_session))[-4:]}.\n"
                        f"User: {http_session.username} is unauthorized. "
                        f"Starting authorization:\n"
                        f"  redirecting to {location:.100}\n"
                        f"  from           {initial_response.url:.100}\n"
                        f"  cookies:\n"
                        f"{self.cookies_repr(4)}")
                    if hasattr(session, "__authorization_started"):
                        delattr(session, "__authorization_started")
                        raise RuntimeError(f"Authorization of {http_session.username} not "
                                           f"successful for session "
                                           f"{str(id(http_session))[-4:]}.")
                    setattr(session, "__authorization_started", next_url.path)
                    if self.CSRF_NAME not in session.cookies and\
                            self.CSRF_NAME in initial_response.cookies:
                        extract_cookies_to_jar(session.cookies,
                                               initial_response.request,
                                               initial_response.raw)
                    oauth2_proxy_csrf_response = session.get(location)
                    oauth2_proxy_csrf_request = oauth2_proxy_csrf_response.request
                    oauth2_proxy_response = session\
                        .post(oauth2_proxy_csrf_request.url,
                              data=self._authorization_credentials_data(),
                              cookies=oauth2_proxy_csrf_response.cookies)
                    if oauth2_proxy_response.ok and self.PROXY_NAME in session.cookies:
                        delattr(session, "__authorization_started")
                        logger.verbose(
                            f"\nLogin hook for session: {str(id(http_session))[-4:]}.\n"
                            f"User: {http_session.username} is authorized. "
                            f"Stopping authorization\n"
                            f"  Cookies:\n"
                            f"{self.cookies_repr(4)}")
                        initial_url = urlparse(initial_response.request.url)
                        if "oauth2/sign_in" not in initial_url.path:
                            return re_send_initial_request(initial_response, session)
                        return oauth2_proxy_response
                else:
                    logger.verbose(
                        f"\nLogin hook for session: {str(id(http_session))[-4:]}.\n"
                        f"User: {http_session.username} is redirected.\n"
                        f"  redirecting to {location:.100}\n"
                        f"  from           {initial_response.url:.100}\n"
                        f"  cookies:\n"
                        f"{self.cookies_repr(4)}")
            return initial_response

        def re_send_initial_request(initial_response: Response, session: Session):
            re_authorized_request = initial_response.request.copy()
            tt_resend_counter = int(re_authorized_request.headers.get("tt_resend_counter", "1"))
            logger.verbose(
                f"\nLogin hook for session: {str(id(http_session))[-4:]}.\n"
                f"User: {http_session.username} is re-sending: {tt_resend_counter}")
            if tt_resend_counter > 5:
                if hasattr(session, "__authorization_started"):
                    delattr(session, "__authorization_started")
                raise RuntimeError("Resend counter exceeded.")
            headers = re_authorized_request.headers
            headers.pop('Cookie', None)
            headers.update({
                "tt_resend_counter": str(tt_resend_counter + 1)
            })
            re_authorized_request.prepare_headers(headers)
            re_authorized_request.prepare_cookies(session.cookies)
            if tt_resend_counter > 1:
                re_send_cookie = re_authorized_request.headers.get("Cookie", None)

                re_send_cookie = re_send_cookie[len("_oauth2_proxy="):][:30] if re_send_cookie \
                    else "None"
                session_cookie = session.cookies.get(self.PROXY_NAME)
                session_cookie = session_cookie[:30] if session_cookie else "None"
                logger.verbose(
                    f"\nLogin hook for session: {str(id(http_session))[-4:]}.\n"
                    f"User: {http_session.username} is re-sending {tt_resend_counter} with:\n"
                    f"{'header cookie:':24} {re_send_cookie}\n"
                    f"{'session cookie:':24} {session_cookie}")
            re_authorized_response = session.send(re_authorized_request)
            return re_authorized_response

        return login_hook_fn

    def authenticate(self) -> AuthBase:
        """Use session credentials to authenticate."""
        session = self.session._session
        logger.verbose(f" ********************* "
                       f"Authenticating {self.session.username} "
                       f"******************")
        if not self.hooks_added:
            logger.verbose("adding hooks")
            session.hooks["response"].append(self.login_hook(self.session))
            self.hooks_added = True
        return OAuth2ProxyAuth()

    @property
    def authenticated(self) -> bool:
        return True

    def parse_params(self, params: dict) -> None:
        """params parser to pass configuration customizations"""
        self.request_params = params

    def _authorization_credentials_data(self):
        """Prepare request data."""
        data = {
            "login": self.session.user.id,
            "password": self.session.user.password,
        }
        return data


class ClientAuthFactory(object):
    """Client authentication factory."""

    EMPTY_URL = ""

    @staticmethod
    def get(username: Union[HttpUser, str] = None,
            password: str = None,
            auth_type: ClientAuthType = None,
            proxies: dict = None,
            cert: tuple = None,
            auth_url: str = EMPTY_URL,
            params: dict = None) -> ClientAuthBase:
        """Create client authentication for given type."""
        session = HttpSession(username, password, proxies, cert)

        if auth_type == ClientAuthType.TOKEN_AUTH:
            return ClientAuthToken(auth_url, session, params)

        elif auth_type == ClientAuthType.HTTP_SESSION:
            return ClientAuthSession(auth_url, session, params)

        elif auth_type == ClientAuthType.HTTP_BASIC:
            return ClientAuthHttpBasic(auth_url, session, params)

        elif auth_type == ClientAuthType.TOKEN_NO_AUTH:
            return ClientAuthTokenProvided(auth_url, session, params)

        elif auth_type == ClientAuthType.NO_AUTH:
            return ClientAuthNoAuth(auth_url, session)

        elif auth_type == ClientAuthType.SSL:
            return ClientAuthSsl(auth_url, session)

        elif auth_type == ClientAuthType.LOGIN_PAGE:
            return ClientAuthLoginPage(auth_url, session, params)

        elif auth_type == ClientAuthType.OAUTH2_PROXY_AUTH:
            return ClientAuthOAuth2Proxy(auth_url, session, params)

        else:
            raise ClientAuthFactoryInvalidAuthTypeException(auth_type)
