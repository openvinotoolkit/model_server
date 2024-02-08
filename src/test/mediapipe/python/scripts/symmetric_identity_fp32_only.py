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
import numpy as np
from pyovms import Tensor
class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        self.model_outputs = kwargs 
        return

    def execute(self, inputs: list, kwargs: dict = {}):
        # Return input with different name. Reject non FP32 tensors.
        outputs = []
        for input in inputs:
            if input.datatype != "FP32":
                raise Exception("Not supported datatype")
            input_npy = np.array(input)
            output_npy = input_npy
            output_name = input.name.replace("input", "output")
            outputs.append(Tensor(output_name, output_npy))
        return outputs

    def finalize(self, kwargs: dict):
        return None