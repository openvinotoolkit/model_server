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

from typing import List

from pyovms import Tensor

# Fetched from:
# https://docs.openvino.ai/nightly/ovms_docs_python_support_reference.html#basic-example

class OvmsPythonModel:
    # Assuming this code is used with nodes
    # that have single input and single output

    def initialize(self, kwargs: dict):
        self.node_name = kwargs["node_name"]
        self.input_names = kwargs["input_names"]
        self.output_names = kwargs["output_names"]

    def execute(self, inputs: list):
        text = [bytes(input).decode() for input in inputs]
        incremented_text = self.increment(text)
        outputs_list = [Tensor(output_name, incremented_text[i]) for i, output_name in enumerate(self.output_names)]
        return outputs_list

    @staticmethod
    def increment(text_input_data: List[bytes]):
        data = [(text * 2).encode() for text in text_input_data]
        return data
