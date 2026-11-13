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

from tests.functional.config import ovms_grpc_port, ovms_rest_port
from tests.functional.utils.inference.communication import GRPC, REST


class OvmsExternal:
    """ Represents an already running OVMS, pointed to by TT_OVMS_REST_PORT/TT_OVMS_GRPC_PORT and TT_SERVER_ADDRESS. """

    def __init__(self, rest_port=ovms_rest_port, grpc_port=ovms_grpc_port):
        self.ovms_ports = {}
        if rest_port is not None:
            self.ovms_ports[REST] = rest_port
        if grpc_port is not None:
            self.ovms_ports[GRPC] = grpc_port

    def get_port(self, api_type):
        if isinstance(api_type, str):
            return self.ovms_ports[api_type]
        return self.ovms_ports[api_type.type]
