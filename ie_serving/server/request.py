#
# Copyright (c) 2019 Intel Corporation
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
from threading import Event


class Request(Event):
    def __init__(self, inference_input):
        super().__init__()
        self.inference_input = inference_input
        self.ireq_index = None
        self.result = None

    def wait_for_result(self):
        super().wait()
        return self.result, self.ireq_index

    def set_result(self, ireq_index, result):
        self.ireq_index = ireq_index
        self.result = result
        super().set()
