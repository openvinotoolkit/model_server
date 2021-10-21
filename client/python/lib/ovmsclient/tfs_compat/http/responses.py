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
from ovmsclient.tfs_compat.base.responses import (PredictResponse, ModelMetadataResponse,
                                                  ModelStatusResponse)


class HttpPredictResponse(PredictResponse):

    def to_dict(self):
        response_json = json.loads(self.raw_response.text)
        outputs = response_json.get("outputs", None)
        if outputs:
            result_dict = {}
            if isinstance(outputs, dict):
                result_dict["outputs"] = {}
                for key, value in outputs.items():
                    result_dict["outputs"][key] = np.array(value)
            else:
                result_dict["outputs"] = np.array(outputs)
            return result_dict
        return response_json


class HttpModelMetadataResponse(ModelMetadataResponse):

    def to_dict(self):
        response_json = json.loads(self.raw_response.text)
        metadata = response_json.get("metadata", None)
        model_spec = response_json.get("modelSpec", None)
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

    def to_dict(self):
        response_json = json.loads(self.raw_response.text)
        model_version_status = response_json.get("model_version_status", None)
        if model_version_status:
            result_dict = {}
            for version_status in model_version_status:
                version = int(version_status["version"])
                result_dict[version] = {
                    "state": version_status["state"],
                    "error_code": version_status["status"]["error_code"],
                    "error_message": version_status["status"]["error_message"]
                }
            return result_dict
        return response_json


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
