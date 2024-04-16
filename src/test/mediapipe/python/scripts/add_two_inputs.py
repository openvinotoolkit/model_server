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
import struct
from pyovms import Tensor
class OvmsPythonModel:

    def execute(self, inputs: list):
        outputs = []
        if len(inputs) != 2:
            raise Exception("Incorrect number of inputs")
        if inputs[0].shape != inputs[1].shape:
            raise Exception("Both input should have same shape")
        if inputs[0].datatype != inputs[1].datatype:
            raise Exception("Both input should have same datatype") 
        dataFormat = ''
        if inputs[0].datatype == "FP32":
            dataFormat = '{}f'.format(inputs[0].shape[1])
        if inputs[0].datatype == "FP64":
            dataFormat = '{}d'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "INT32":
            dataFormat = '{}i'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "UINT32":
            dataFormat = '{}I'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "INT8":
            dataFormat = '{}b'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "UINT8":
            dataFormat = '{}B'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "INT64":
            dataFormat = '{}q'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "UINT64":
            dataFormat = '{}Q'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "BOOL":
            dataFormat = '{}?'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "INT16":
            dataFormat = '{}h'.format(inputs[0].shape[1])
        elif inputs[0].datatype == "UINT16":
            dataFormat = '{}H'.format(inputs[0].shape[1])
        input_1 = struct.unpack(dataFormat, bytes(inputs[0]))
        input_2 = struct.unpack(dataFormat, bytes(inputs[1]))
        output_array = list(map(sum, zip(input_1, input_2))) 
        output_data = struct.pack(dataFormat, *output_array)
        output_name = "out"
        outputs.append(Tensor(output_name, output_data, shape=inputs[0].shape, datatype = inputs[0].datatype))
        return outputs