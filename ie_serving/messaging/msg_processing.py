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

from grpc import StatusCode

def extract_ipc_response(ipc_endpoint_response, expected_response_type):
    error_code = StatusCode.OK
    error_message = None
    ipc_response = None
    if ipc_endpoint_response.HasField(expected_response_type):
        ipc_response = getattr(ipc_endpoint_response, expected_response_type)
    elif ipc_endpoint_response.HasField("error_response"):
        error_code = ipc_endpoint_response.error_response.error_code
        error_message = ipc_endpoint_response.error_response.error_message
    else:
        # Internal server error on unexpected endpoint response content
        error_code = StatusCode.INTERNAL
        error_message = "Unexpected IPC message exchanged between components"
    status = {"error_code": error_code, "error_message": error_message}
    return ipc_response, status