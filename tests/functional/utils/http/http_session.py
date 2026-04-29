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

"""Wrapper for the Session class from the requests library."""

import json
from abc import ABCMeta
from http.cookiejar import Cookie, CookieJar
from typing import Optional, Union

from requests import PreparedRequest, Request, Session, exceptions
from requests.adapters import HTTPAdapter
from requests_toolbelt import MultipartEncoder
from retry import retry

from tests.functional.utils.assertions import UnexpectedResponseError
from tests.functional.utils.http.base import HttpMethod, HttpUser
from tests.functional.utils.logger import LoggerType, get_logger
from tests.functional.config import http_proxy, https_proxy, logged_response_body_length, no_proxy, ssl_validation

POOL_SIZE = 100

logger = get_logger(__name__)


class HttpSession(object, metaclass=ABCMeta):
    """HttpSession is wrapper for the Session class from the requests library.

    It stores the information about the username, password, possible proxies and certificates.
    It does not do any authentication per se.
    For ease of use it has a "request" method that prepares and performs the request.
    """
    def __init__(self, username: Union[HttpUser, str] = None, password: str = None,
                 proxies: dict = None, cert: tuple = None):
        self._user = username if isinstance(username, HttpUser) else HttpUser(username, password)
        self._session = Session()
        if proxies is not None:
            self._session.proxies = proxies
        elif http_proxy:
            self._session.proxies = {"http": http_proxy,
                                     "https": https_proxy,
                                     "no_proxy": no_proxy}

        self._session.verify = ssl_validation
        if cert is not None:
            self._session.cert = cert
            try:
                self._session.verify = cert[2]
            except IndexError:
                pass

        adapter = HTTPAdapter(pool_connections=POOL_SIZE, pool_maxsize=POOL_SIZE)
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

    @property
    def user(self):
        return self._user

    @property
    def username(self) -> str:
        """Session user name."""
        return self.user.name

    @property
    def password(self) -> str:
        """Session user password."""
        return self.user.password

    @property
    def session(self) -> Session:
        """Session cookies."""
        return self._session

    @property
    def cookies(self):
        """Session cookies."""
        return self.session.cookies

    def get_cookie(self, name: str) -> Optional[Cookie]:
        return self.get_cookie_from_jar(name, self.cookies)

    @staticmethod
    def get_cookie_from_jar(name: str, cookies: CookieJar) -> Optional[Cookie]:
        matching_cookies = filter(lambda c: c.name == name,
                                  iter(cookies))
        first_cookie = next(matching_cookies, None)
        for cookie in matching_cookies:
            logger.warning(f"Additional matching cookie: {cookie}")
        return first_cookie

    # pylint: disable=too-many-arguments
    def request(self, method: HttpMethod, url, path="", path_params=None,
                headers=None, files=None, data=None, params=None, auth=None,
                body=None, log_message="", raw_response=False, timeout=None,
                raise_exception=True, log_response_content=True):
        """Wrapper for the request method from the Session library"""
        path_params = {} if path_params is None else path_params
        url = f"{url}/{path}".format(**path_params)
        logger.debug(f"\nSending rq for session: {str(id(self))[-4:]} "
                    f"request {method} as a\n"
                    f"user: {self.username} to {url}")
        request = self._request_prepare(method, url, headers, files,
                                        data, params, auth, body, log_message)
        return self._request_perform(request, path, path_params, raw_response, timeout=timeout,
                                     raise_exception=raise_exception,
                                     log_response_content=log_response_content)

    # pylint: disable=too-many-arguments
    def _request_prepare(self, method, url, headers, files, data, params, auth, body, log_message):
        """Prepare request to perform."""
        request = Request(method=method, url=url, headers=headers,
                          files=files, data=data, params=params,
                          auth=auth, json=body)

        prepared_request = self._session.prepare_request(request)
        logger.debug(f"Prepared request: {prepared_request.__dict__}")
        return prepared_request

    def _request_perform(self, request: PreparedRequest, path: str, path_params, raw_response: bool,
                         timeout: int, raise_exception: bool, log_response_content=True):
        """Perform request and return response."""
        response = self._send_request_and_get_raw_response(request, timeout=timeout)
        if log_response_content:
            # workaround for downloading large files - reading response.text takes too long
            HttpSession.log_http_response(response)
        else:
            HttpSession.log_http_response(response, logged_body_length=0)

        if raise_exception and not response.ok and response.text.strip() != "session_expired":
            raise UnexpectedResponseError(response.status_code, response.text)

        if raw_response is True:
            return response

        try:
            return json.loads(response.text)
        except ValueError:
            return response.text

    @retry(exceptions=exceptions.ConnectionError, tries=2)
    def _send_request_and_get_raw_response(self, request: PreparedRequest, timeout: int):
        return self._session.send(request, timeout=timeout)

    @staticmethod
    def _format_message_body(msg_body, logged_body_length=None):
        limit = int(logged_response_body_length) if logged_body_length is None else logged_body_length
        if 0 < limit < len(msg_body):
            half = limit // 2
            msg_body = f"{msg_body[:half]}[...]{msg_body[-half:]}"
        elif limit == 0:
            msg_body = "[...]"
        return msg_body

    @staticmethod
    def log_http_response(response, logged_body_length=None, history_depth=1):
        """If logged_body_length < 0, full response body is logged"""
        if history_depth > 0 and response.history:
            for history_response in response.history:
                HttpSession.log_http_response(response=history_response,
                                  logged_body_length=logged_body_length,
                                  history_depth=history_depth - 1)

        msg_body = HttpSession._format_message_body(response.text, logged_body_length)
        msg = [
            "\n----------------Response------------------",
            f"Status code: {response.status_code}",
            f"Headers: {response.headers}",
            f"Content: {msg_body}",
            "-----------------------------------------\n"
        ]
        get_logger(LoggerType.HTTP_RESPONSE).debug("\n".join(msg))

    @staticmethod
    def log_http_request(prepared_request, username, description="", data=None):
        if isinstance(prepared_request.body, MultipartEncoder):
            body = prepared_request.body
        else:
            prepared_body = prepared_request.body
            body = prepared_body if not data else json.dumps(data)
        content_type = prepared_request.headers.get("content-type", "")
        msg_body = HttpSession._format_message_body(body) if (
                body is not None and "multipart/form-data" not in content_type
        ) else ""
        msg = [
            description,
            "----------------Request------------------",
            f"Client name: {username}",
            f"URL: {prepared_request.method} {prepared_request.url}",
            f"Headers: {prepared_request.headers}",
            f"Body: {msg_body}",
            "-----------------------------------------"
        ]
        get_logger(LoggerType.HTTP_REQUEST).debug("\n".join(msg))
