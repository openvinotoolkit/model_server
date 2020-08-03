#
# Copyright (c) 2020 Intel Corporation
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

import os
import errno
import re
import socket
from datetime import datetime

from utils.helpers import SingletonMeta


def get_ports_prefixes():
    ports_prefixes = os.environ.get("PORTS_PREFIX", "90 55")
    grpc_ports_prefix, rest_ports_prefix = [
        port_prefix for port_prefix in ports_prefixes.split(" ")]
    return {"grpc_ports_prefix": grpc_ports_prefix,
            "rest_ports_prefix": rest_ports_prefix}


def get_ports_suffix():
    suf = Suffix()
    suffix = str(suf.index) if len(str(suf.index)) == 2 else "0" + str(suf.index)
    suf.index += 1
    return suffix


class TestsSuffix(metaclass=SingletonMeta):
    string = None


def get_tests_suffix():
    tests_suffix = TestsSuffix()
    if not tests_suffix.string:
        tests_suffix.string = os.environ.get("TESTS_SUFFIX", generate_test_object_name(prefix="suffix"))
    return tests_suffix.string


def get_ports_for_fixture():
    ports_prefixes = get_ports_prefixes()

    port_found = False
    while not port_found:
        port_suffix = get_ports_suffix()

        grpc_port = ports_prefixes["grpc_ports_prefix"] + port_suffix
        rest_port = ports_prefixes["rest_ports_prefix"] + port_suffix

        location_grpc = ("", int(grpc_port))
        location_rest = ("", int(rest_port))
        try:
            sock_grpc = socket.socket()
            sock_grpc.bind(location_grpc)

            sock_rest = socket.socket()
            sock_rest.bind(location_rest)

        except socket.error as e:
            if e.errno != errno.EADDRINUSE:
                raise Exception("Not expected exception found "
                                "while getting ports for fixture {}:".format(e))
            # Other error means address is in use and we must proceed
            # to the next candidate
            continue

        # No exception raised - port is available.
        port_found = True

    return grpc_port, rest_port


class Suffix(metaclass=SingletonMeta):
    index = 0


class ObjectName:
    _date_format, _time_format, _ms_format = "%d", "%H%M%S", "%f"
    _NON_ALPHA_NUM = r'[^a-z0-9]+'

    def __init__(self, short=False, prefix=None, separator='_'):
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
        hostname = socket.gethostname().split(".", 1)[0].lower()
        self._prefix = prefix if prefix else hostname
        self._prefix = "{}{}".format(worker_id, self._prefix)
        self._separator = separator
        self._short = short
        self._now = datetime.now()

    def __str__(self):
        separator = '' if self._short else self._separator
        parts = [self._prefix] + self.stem
        name = separator.join(parts).lower()
        return re.sub(self._NON_ALPHA_NUM, separator, name)

    @property
    def stem(self) -> list:
        seed = [
            self._now.strftime(self._date_format),
            self._now.strftime(self._time_format),
            self._now.strftime(self._ms_format)
        ]
        return seed[:2] if self._short else seed

    def build(self) -> str:
        return str(self)


def generate_test_object_name(short=False, prefix=None, separator="_"):
    name = ObjectName(short=short, prefix=prefix, separator=separator)
    return name.build()
