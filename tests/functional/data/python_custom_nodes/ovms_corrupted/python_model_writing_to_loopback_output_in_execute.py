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


class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        self.node_name = kwargs["node_name"]
        self.input_names = kwargs["input_names"]
        self.output_names = ["loopback"]
        self.class_methods = {
            name: func for name, func in inspect.getmembers(OvmsPythonModel, predicate=inspect.isfunction)
        }
        print(f'kwargs in writing to loopback: {kwargs["output_names"]}')

    def execute(self, inputs: list):
        input_data = inputs[0]
        text = bytes(input_data).decode()

        # Expected method should have the same name as self.node_name.
        func = self.class_methods.get(self.node_name, None)
        assert func is not None, f"Function for {self.node_name} is None: {func}"

        # Create Tensor with encoded text.
        # Should be consistent with the value set in PythonCalculator.
        # A list of Tensors is expected, even if there's only one output.
        from pyovms import Tensor

        for i in range(len(text)):
            output_data = func(self, text=text, counter=i)
            outputs_list = [Tensor(output, output_data) for output in self.output_names]
            yield outputs_list

    def loopback_upper_text(self, text: bytes, counter):
        data = text[:counter] + text[counter].upper() + text[counter + 1 :]
        return data.encode()
