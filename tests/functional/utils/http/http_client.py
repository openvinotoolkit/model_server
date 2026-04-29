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

from tests.functional.utils.http.base import HttpMethod
from tests.functional.utils.http.client_auth.base import ClientAuthBase
from tests.functional.utils.http.http_session import HttpSession


class HttpClient(object):
    """Http api client."""

    def __init__(self, url: str, auth: ClientAuthBase):
        self.url = url
        self._auth = auth

    @property
    def auth(self) -> ClientAuthBase:
        """Client client_auth."""
        return self._auth

    @property
    def cookies(self):
        """Session cookies."""
        return self.auth.session.cookies

    @property
    def session(self) -> HttpSession:
        return self._auth.session

    @session.setter
    def session(self, session: HttpSession):
        self._auth.session = session

    def get_cookie(self, name: str):
        return self.session.get_cookie(name)

    # pylint: disable=too-many-arguments
    def request(self, method: HttpMethod, path, path_params=None, url=None, headers=None, files=None, params=None,
                data=None, body=None, credentials=None, msg="", raw_response=False, timeout=900, raise_exception=True,
                log_response_content=True):
        """Perform request and return response."""
        if not self._auth.authenticated:
            self._auth.authenticate()
        if credentials is None:
            credentials = self._auth.http_auth
        url = self.url if url is None else url
        response = self._auth.session.request(
            method=method,
            url=url,
            path=path,
            path_params=path_params,
            headers=headers,
            files=files,
            params=params,
            data=data,
            body=body,
            auth=credentials,
            log_message=msg,
            raw_response=raw_response,
            timeout=timeout,
            raise_exception=raise_exception,
            log_response_content=log_response_content
        )

        if "session_expired" == format(response).strip():
            self._auth.authenticate()
            return self._auth.session.request(
                method=method,
                url=url,
                path=path,
                path_params=path_params,
                headers=headers,
                files=files,
                params=params,
                data=data,
                body=body,
                auth=self._auth.http_auth,
                log_message=msg,
                raw_response=raw_response,
                timeout=timeout,
                raise_exception=raise_exception
            )

        return response
