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

import re
import socket
from time import time


class HttpSocketWrapper:
    _EXP_BODY_LEN = re.compile(r'Content-Length: (\d+)\s')
    _EXP_BODY_SEPARATOR = re.compile(r'\s{4}')

    def __init__(self, host, port):
        self._family = socket.AF_INET
        self._type = socket.SOCK_STREAM
        self.host = host
        self.port = port

    def send(self, method, path, body='', headers={}):
        default_headers = {"Host": f"{self.host}:{self.port}",
                           "Content-Length": str(len(body)),
                           "Connection": 'close'}
        default_headers.update(headers)
        header = f"{method} {path} HTTP/1.1\n"
        for name, value in default_headers.items():
            header += f"{name}: {value}\n"
        header += "\n"
        msg = header.encode('ascii') + body.encode('ascii')

        with socket.socket(self._family, self._type) as soc:
            soc.connect((self.host, self.port))
            start = time()
            soc.sendall(msg)

            header, body = self._receive(soc)
            end = time()

        return end - start, header, body

    def _receive(self, soc):
        base_len, header, body = 10, '', ''

        while HttpSocketWrapper._EXP_BODY_SEPARATOR.search(header) is None:
            header += soc.recv(base_len).decode('ascii')

        match = HttpSocketWrapper._EXP_BODY_SEPARATOR.search(header)
        body = header[match.end():]
        header = header[:match.start()]

        match = HttpSocketWrapper._EXP_BODY_LEN.search(header)
        if match:
            body_len = int(match.group(1))
            base_len = body_len - len(body)
            body += soc.recv(base_len).decode('ascii')
        else:
            tmp_msg = soc.recv(base_len).decode('ascii')
            while len(tmp_msg) > 0:
                body += tmp_msg
                tmp_msg = soc.recv(base_len).decode('ascii')

        return header, body
