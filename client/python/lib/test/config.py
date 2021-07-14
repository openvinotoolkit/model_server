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

MODEL_SPEC_VALID = [
    ("model_name", 1),
    ("3", 17),
    ("*.-", 0),
    ("model_name", 9223372036854775806),
]

MODEL_SPEC_INVALID = [
    (12, 1, TypeError, "model_name type should be string, but is int"),
    (None, -1, TypeError, "model_name type should be string, but is NoneType"),
    (None, 1, TypeError, "model_name type should be string, but is NoneType"),
    ("model_name", "3", TypeError, "model_version type should be int, but is str"),
    ("model_name" , None, TypeError, "model_version type should be int, but is NoneType"),
    ("model_name", -1, ValueError, f'model_version should be in range <0, {2**63-1}>'),
    ("model_name", 9223372036854775809, ValueError, f'model_version should be in range <0, {2**63-1}>'),
]

from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusResponse, ModelVersionStatus
from tensorflow_serving.util.status_pb2 import StatusProto

def create_model_status_response(model_version, error_code, error_message, model_state):
    status = StatusProto()
    status.error_code = error_code
    status.error_message = error_message

    model_version_status = ModelVersionStatus()
    model_version_status.version = model_version
    model_version_status.state = model_state
    model_version_status.status.CopyFrom(status)

    return model_version_status

def merge_model_status_responses(responses):
    raw_response = GetModelStatusResponse()
    model_versions = [response for response in responses]
    raw_response.model_version_status.extend(model_versions)

    return raw_response

RAW_MODEL_RESPONSE = {
    1 : (1, 1, "error" , 10),
    2 : (2, 2, "error2", 20),
    3 : (3, 3, "error3", 30)
}

RAW_MODEL_RESPONSE_VALID = [
    ([RAW_MODEL_RESPONSE[1]]),
    ([RAW_MODEL_RESPONSE[2]]),
    ([RAW_MODEL_RESPONSE[1], RAW_MODEL_RESPONSE[2], RAW_MODEL_RESPONSE[3]]),
]
