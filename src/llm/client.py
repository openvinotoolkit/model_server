#
# Copyright (c) 2024 Intel Corporation
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

# THIS is a temporary script for testing llm engine
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("localhost:9000")
prompt = "What is OpenVINO?"
infer_input = grpcclient.InferInput("prompt", [1], "BYTES")
infer_input._raw_content = prompt.encode("ascii")
results = client.infer("llm_graph", [infer_input])
print(results._result.raw_output_contents[0])
