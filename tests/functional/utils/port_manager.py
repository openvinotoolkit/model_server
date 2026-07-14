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

import errno
import psutil
import socket
import threading

from tests.functional.utils.core import NamedSingletonMeta
from tests.functional.utils.logger import get_logger
from tests.functional.utils.helpers import get_xdist_worker_count, get_xdist_worker_nr
from tests.functional.constants.os_type import OsType, get_host_os

logger = get_logger(__name__)


class PortManager(metaclass=NamedSingletonMeta):
    _thread_lock = threading.Lock()

    def __init__(self, name: str, starting_port: int = None, pool_size: int = None):
        self.name = name
        assert starting_port is not None, f"Lack of starting port while creating instance {name} of PortManager"
        assert pool_size is not None, f"Lack of pool size while creating instance {name} of PortManager"
        assert pool_size > 0, f"Not expected pool size given for manager {name}, should be > 0."

        self.xdist_worker_count = get_xdist_worker_count()
        self.xdist_current_worker = get_xdist_worker_nr()

        self.starting_port = int(starting_port + (pool_size / self.xdist_worker_count) * self.xdist_current_worker)
        self.pool_size = int(pool_size / self.xdist_worker_count)

        self.reserved_ports = []
        self.allowed_ports = list(range(self.starting_port, self.starting_port + self.pool_size))
        self.check_allowed_ports()

    def check_allowed_ports(self):
        unavailable_ports = []
        if get_host_os() == OsType.Windows:
            for port in self.allowed_ports:
                for conn in psutil.net_connections():
                    if conn.laddr.port == port:
                        unavailable_ports.append(port)
                        break
        else:
            for port in self.allowed_ports:
                try:
                    sock = socket.socket()
                    sock.bind(('', port))
                    sock.close()
                except socket.error as exc:
                    if exc.errno != errno.EADDRINUSE:
                        raise Exception(f"Not expected exception found in port manager {self.name}: {exc}")
                    unavailable_ports.append(port)
        if unavailable_ports:
            logger.warning(f"Unavailable ports found: {unavailable_ports}")

    def get_port(self):
        with self._thread_lock:
            logger.debug(
                f"Getting port for Port Manager: {self.name}\n"
                f"allowed ports: {', '.join([str(port) for port in self.allowed_ports])}\n"
                f"reserved ports: {', '.join([str(port) for port in self.reserved_ports])}"
            )
            for port in self.allowed_ports[:]:
                generated_port = self.reserve_port(port=port)
                logger.debug(f"Generated port for Port Manager {self.name}: {generated_port}")
                if generated_port:
                    logger.debug(f"Reserved port for Port Manager {self.name}: {generated_port}")
                    return generated_port
            else:
                raise Exception(f"Ports pool {self.name} has been used up. "
                                "Consider releasing ports or increase the pool size.")

    def reserve_port(self, port):
        if get_host_os() == OsType.Windows:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    break
            else:
                self.reserved_ports.append(port)
                self.allowed_ports.remove(port)
                return port
        else:
            try:
                sock = socket.socket()
                sock.bind(('', port))
                sock.close()
                self.reserved_ports.append(port)
                self.allowed_ports.remove(port)
                return port
            except socket.error as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise Exception(f"Not expected exception found in port manager {self.name}: {exc}")

        self.allowed_ports.remove(port)
        return 0

    def release_port(self, port: int):
        logger.debug(f"Releasing port {port} from Port Manager:")

        with self._thread_lock:
            self.reserved_ports.remove(port)
            self.allowed_ports.append(port)
