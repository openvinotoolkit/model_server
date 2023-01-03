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

import json
import numpy as np
from ovmsclient.tfs_compat.protos.tensorflow.core.protobuf.error_codes_pb2 import Code as ErrorCode
from ovmsclient.tfs_compat.base.responses import (PredictResponse, ModelMetadataResponse,
                                                  ModelStatusResponse)
from ovmsclient.tfs_compat.base.errors import raise_from_http_response


class HttpPredictResponse(PredictResponse):

    def to_dict(self):
        response_json = json.loads(self.raw_response.text)
        error_message = response_json.get("error", None)
        outputs = response_json.get("outputs", None)

        if not error_message and not outputs:
            raise (ValueError("No outputs or error found in response"))

        if error_message:
            raise_from_http_response(self.raw_response.status_code, error_message)

        result_dict = {"outputs": {}}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                result_dict["outputs"][key] = np.array(value)
        else:
            # For models with only one output, put results directly under "outputs" key
            result_dict["outputs"] = np.array(outputs)
        return result_dict


class HttpModelMetadataResponse(ModelMetadataResponse):

    def to_dict(self):
        response_json = json.loads(self.raw_response.text)
        metadata = response_json.get("metadata", None)
        model_spec = response_json.get("modelSpec", None)
        error_message = response_json.get("error", None)

        if error_message:
            raise_from_http_response(self.raw_response.status_code, error_message)

        if metadata and model_spec:
            io_info = metadata["signature_def"]["signatureDef"]["serving_default"]
            result_dict = {}
            for type in ["inputs", "outputs"]:
                result_dict[type] = {}
                for name, raw_info in io_info[type].items():
                    dtype = raw_info["dtype"]
                    shape = [int(dim["size"]) for dim in raw_info["tensorShape"]["dim"]]
                    result_dict[type][name] = {"dtype": dtype, "shape": shape}
            result_dict["model_version"] = int(model_spec["version"])
            return result_dict
        return response_json


class HttpModelStatusResponse(ModelStatusResponse):

    # Error codes in REST API are returned as strings.
    # To unify the return value between gRPC and HTTP we map error codes strings to numbers
    # to match gRPC response.
    _ERROR_CODE_TO_NUMBER = {
        "OK": ErrorCode.OK,
        "CANCELLED": ErrorCode.CANCELLED,
        "UNKNOWN": ErrorCode.UNKNOWN,
        "INVALID_ARGUMENT": ErrorCode.INVALID_ARGUMENT,
        "DEADLINE_EXCEEDED": ErrorCode.DEADLINE_EXCEEDED,
        "NOT_FOUND": ErrorCode.NOT_FOUND,
        "ALREADY_EXISTS": ErrorCode.ALREADY_EXISTS,
        "UNAUTHENTICATED": ErrorCode.UNAUTHENTICATED,
        "RESOURCE_EXHAUSTED": ErrorCode.RESOURCE_EXHAUSTED,
        "FAILED_PRECONDITION": ErrorCode.FAILED_PRECONDITION,
        "ABORTED": ErrorCode.ABORTED,
        "OUT_OF_RANGE": ErrorCode.OUT_OF_RANGE,
        "UNIMPLEMENTED": ErrorCode.UNIMPLEMENTED,
        "INTERNAL": ErrorCode.INTERNAL,
        "UNAVAILABLE": ErrorCode.UNAVAILABLE,
        "DATA_LOSS": ErrorCode.DATA_LOSS,
        "DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_":
            ErrorCode.DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_
    }

    def to_dict(self):
        response_json = json.loads(self.raw_response.text)
        error_message = response_json.get("error", None)
        model_version_status = response_json.get("model_version_status", None)
        if not error_message and not model_version_status:
            raise (ValueError("No model status or error found in response"))

        if error_message:
            raise_from_http_response(self.raw_response.status_code, error_message)

        result_dict = {}
        for version_status in model_version_status:
            version = int(version_status["version"])
            error_code = self._ERROR_CODE_TO_NUMBER.get(version_status["status"]["error_code"],
                                                        ErrorCode.UNKNOWN)
            result_dict[version] = {
                "state": version_status["state"],
                "error_code": error_code,
                "error_message": version_status["status"]["error_message"]
            }
        return result_dict


class HttpConfigStatusResponse:

    def to_dict(self):
        '''
        Return status in dictionary format:

        .. code-block::

            {
                ...
                <model_name>: {
                    ...
                    <version_number>: {
                        "state": <model_version_state>,
                        "error_code": <error_code>,
                        "error_message": <error_message>
                    },
                    ...
                },
                ...
            }

        '''
        pass
