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


def get_tests_suffix():
    return os.environ.get("TESTS_SUFFIX", "default")


def get_ports_for_fixture():
    ports_prefixes = get_ports_prefixes()
    port_suffix = get_ports_suffix()
    grpc_port = ports_prefixes["grpc_ports_prefix"]+port_suffix
    rest_port = ports_prefixes["rest_ports_prefix"]+port_suffix
    return grpc_port, rest_port


class Suffix(metaclass=SingletonMeta):
    index = 0
