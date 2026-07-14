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

class OvmsPythonModel:

    EXCEPTION_INITIALIZE = "exception_initialize"
    EXCEPTION_EXECUTE = "exception_execute"
    EXCEPTION_FINALIZE = "exception_finalize"

    def initialize(self, kwargs: dict):
        self.node_name = kwargs["node_name"]
        self.input_names = kwargs["input_names"]
        self.output_names = kwargs["output_names"]
        if self.node_name == self.EXCEPTION_INITIALIZE:
            raise Exception(self.EXCEPTION_INITIALIZE)

    def execute(self, inputs: list):
        if self.node_name == self.EXCEPTION_EXECUTE:
            raise Exception(self.EXCEPTION_EXECUTE)
        input_data = inputs[0]
        text = bytes(input_data).decode()
        output_data = text.upper().encode()
        # Create Tensor with encoded text.
        # Should be consistent with the value set in PythonCalculator.
        # A list of Tensors is expected, even if there's only one output.
        from pyovms import Tensor
        outputs_list = [Tensor(output, output_data) for output in self.output_names]
        return outputs_list

    def finalize(self):
        if self.node_name == self.EXCEPTION_FINALIZE:
            raise Exception(self.EXCEPTION_FINALIZE)
