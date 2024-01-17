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
        self.output_name = kwargs["output_names"][0]

    def execute(self, inputs: list, kwargs: dict = {}):
        # Increment every element of every input and return them with changed tensor name.
        input = inputs[0]
        input_npy = np.array(input)
        print("input_npy:" + str(input_npy))
        print("input_npy.dtype:" + str(input_npy.dtype))
        output_npy = input_npy + 1
        return [Tensor(self.output_name, output_npy)]
