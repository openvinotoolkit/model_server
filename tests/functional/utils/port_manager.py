#
# Copyright (c) 2022 Intel Corporation
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
import logging
import socket
import errno

from utils.helpers import get_xdist_worker_count, get_xdist_worker_nr

logger = logging.getLogger(__name__)

class SimplePortManager:
    def __init__(self, name: str):
        self.name = name

    def get_port(self):
        generated_port = self.reserve_port()
        logger.debug(f"Generated_port={generated_port}")
        return generated_port

    def reserve_port(self):
        # Idea found in SO:
        # https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
        try:
            sock = socket.socket()
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()
            logger.debug(f"Reserved port={port}")
            return port
        except socket.error as e:
            if e.errno != errno.EADDRINUSE:
                # All ports pool used?
                raise Exception("Not expected exception found in port manager {}: {}".format(self.name, e))
        return 0

    def release_port(self, port: int):
        logger.debug(f"Release port={port}")
        try:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', port))
            sock.close()
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                raise Exception("Address has not been deleted for port manager {}".format(self.name))
            else:
                raise Exception("Not expected exception found in port manager {}: {}".format(self.name, e))
