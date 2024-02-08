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
import struct
from pyovms import Tensor
class OvmsPythonModel:

    def execute(self, inputs: list):
        # Increment every element of every input and return them with changed tensor name.
        outputs = []
        for input in inputs:
            output_name = input.name.replace("input", "output")
            input_fp32 = struct.unpack('f', bytes(input))[0]
            input_fp32 = input_fp32 + 1.0
            output_data = struct.pack('f', input_fp32)
            outputs.append(Tensor(output_name, output_data))
        return outputs
