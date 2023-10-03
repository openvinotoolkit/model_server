#*****************************************************************************
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************
class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        return False
    
    def execute(self, inputs: list, kwargs: dict) -> list:
        # This method will be called for every request.
        # It expects a list of inputs (our custom python objects).
        # It also expects keyword arguments. They will be provided by the calculator to enable advanced processing and flow control.
        # Detailed spec to be provided. 
        #
        # It will returns list of outputs (also our custom python objects).
        ...
        return outputs

    def finalize(self, kwargs: dict):
        # This method will be called once during deinitialization. 
        # It expects keyword arguments. They will map node configuration from pbtxt including node options, node name etc.
        # Detailed spec to be provided. 
        ...
        return None