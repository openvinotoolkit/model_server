#
# Copyright (c) 2026 Intel Corporation
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

import inspect
from typing import List

# Fetched from:
# https://raw.githubusercontent.com/openvinotoolkit/model_server/main/docs/python_support/quickstart.md

class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        self.node_name = kwargs["node_name"]
        self.input_names = kwargs["input_names"]
        self.output_names = kwargs["output_names"]
        self.class_methods = {
            name: func for name, func in inspect.getmembers(OvmsPythonModel, predicate=inspect.isfunction)
        }
        
    def execute(self, inputs: list):
        text_input_data = [bytes(input).decode() for input in inputs]

        # Expected method should have the same name as self.node_name.
        func = self.class_methods.get(self.node_name, None)
        assert func is not None, f"Function for {self.node_name} is None: {func}"
        output_data = func(self, text_input_data=text_input_data)

        # Create Tensor with encoded text.
        # Should be consistent with the value set in PythonCalculator.
        # A list of Tensors is expected, even if there's only one output.
        from pyovms import Tensor
        outputs_list = [Tensor(output_name, output_data[i]) for i, output_name in enumerate(self.output_names)]
        return outputs_list

    def upper_text(self, text_input_data: List[bytes]):
        data = [text.upper().encode() for text in text_input_data]
        return data
