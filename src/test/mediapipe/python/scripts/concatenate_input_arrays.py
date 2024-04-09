#*****************************************************************************
# Copyright 2024 Intel Corporation
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

    def initialize(self, kwargs):
        self.model_outputs = {}

    def execute(self, inputs: list, kwargs: dict = {}):
        # Increment every element of every input and return them with changed tensor name.
        outputs = []
        output_npy = np.array([[]])
        for input in inputs:
            input_npy = np.array(input)
            print(input_npy)
            if input_npy.dtype != np.float32:
                raise Exception("input type should be np.float32") 
            output_npy = np.hstack([output_npy, input_npy])
        output_name = "output"
        output_npy = output_npy.astype(np.float32)
        outputs.append(Tensor(output_name, output_npy))
        return outputs
