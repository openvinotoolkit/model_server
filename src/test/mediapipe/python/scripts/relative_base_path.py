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
import os

class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        base_path = kwargs['base_path']
        print("Base path: " + os.path.normpath(base_path) + "; Expected: " + os.path.normpath(os.path.dirname(os.path.realpath(__file__))), flush=True)
        # On Windows, paths might slightly differ in terms of letter case, so we make comparison case-insensitive
        assert os.path.normpath(base_path).lower() == os.path.normpath(os.path.dirname(os.path.realpath(__file__))).lower()
        return

    def execute(self, inputs: dict) -> bool:
        return None
    
    def finalize(self, kwargs: dict):
        return
