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

from ovmsclient.tfs_compat.base.responses import PredictResponse, ModelMetadataResponse, ModelStatusResponse

class GrpcPredictResponse(PredictResponse):
    pass

class GrpcModelMetadataResponse(ModelMetadataResponse):
    
    def to_dict(self):
        raise NotImplementedError

class GrpcModelStatusResponse(ModelStatusResponse):
    
    _STATE_TO_STRING_MAPPING = {
        0: "UNKNOWN",
        10: "START",
        20: "LOADING",
        30: "AVAILABLE",
        40: "UNLOADING",
        50: "END"
        }

    def to_dict(self):
        result_dictionary = {}
        model_version_status = self.raw_response.model_version_status
        for model_version in model_version_status:
            result_dictionary[model_version.version] = dict([
                ('state', self._STATE_TO_STRING_MAPPING[model_version.state]),
                ('error_code', model_version.status.error_code),
                ('error_message', model_version.status.error_message),
            ])
        return result_dictionary
