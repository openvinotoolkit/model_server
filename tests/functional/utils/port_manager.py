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
import logging
import socket
import errno

from tests.functional.utils.helpers import get_xdist_worker_count, get_xdist_worker_nr

logger = logging.getLogger(__name__)

class PortManager():

    def __init__(self, name: str, starting_port: int = None, pool_size: int = None):
        self.name = name
        assert starting_port is not None, "Lack of starting port while creating instance {} of PortManager".format(name)
        assert pool_size is not None, "Lack of pool size while creating instance {} of PortManager".format(name)
        assert pool_size > 0, "Not expected pool size given for manager {}, should be > 0.".format(name)

        self.xdist_worker_count = get_xdist_worker_count()
        self.xdist_current_worker = get_xdist_worker_nr()

        self.starting_port = int(starting_port + (pool_size / self.xdist_worker_count) * self.xdist_current_worker)
        self.pool_size = int(pool_size / self.xdist_worker_count)

        self.reserved_ports = []
        self.allowed_ports = list(range(self.starting_port, self.starting_port + self.pool_size))

    def get_port(self):
        logger.debug("Getting port for Port Manager: {}\nallowed ports: {}\nreserved ports: {}"
                     .format(self.name,
                             ", ".join([str(port) for port in self.allowed_ports]),
                             ", ".join([str(port) for port in self.reserved_ports])))
        for port in self.allowed_ports[:]:
            generated_port = self.reserve_port(port=port)
            logger.debug("Generated port for Port Manager {}: {}".format(self.name, generated_port))
            if generated_port:
                logger.debug("Reserved port for Port Manager {}: {}".format(self.name, generated_port))
                return generated_port
        else:
            raise Exception("Ports pool {} has been used up. "
                            "Consider release ports or increase pool size.".format(self.name))

    def reserve_port(self, port):
        try:
            sock = socket.socket()
            sock.bind(('', port))
            sock.close()
            self.reserved_ports.append(port)
            self.allowed_ports.remove(port)
            return port

        except socket.error as e:
            if e.errno != errno.EADDRINUSE:
                raise Exception("Not expected exception found in port manager {}: {}".format(self.name, e))

        self.allowed_ports.remove(port)
        return 0

    def release_port(self, port: int):
        logger.debug("Releasing port for Port Manager: {}\nport to release: {}\nallowed ports: {}\nreserved ports: {}"
                     .format(self.name, port,
                             ", ".join([str(port) for port in self.allowed_ports]),
                             ", ".join([str(port) for port in self.reserved_ports])))
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
        self.reserved_ports.remove(port)
        self.allowed_ports.append(port)
