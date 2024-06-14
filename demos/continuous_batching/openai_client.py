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

from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)
model = "meta-llama/Meta-Llama-3-8B-Instruct"

print("\n*****Unary completions")
response = client.completions.create(
    model=model,
    prompt="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\\Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    max_tokens=100,
    stream=False
)

print("response:",response.choices[0].text)

print("\n****Stream completions")
stream = client.completions.create(
    model=model,
    prompt="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\\Hello. What is OpenVINO?<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    max_tokens=100,
    stream=True
)
print("stream response:")
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="", flush=True)

print("\n****Unary chat completions")
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=100,
    stream=False
)
print("response",response)

print("\n****Stream chat completions")
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "how do you say hello in chinese?"}],
    max_tokens=100,
    stream=True
)
print("stream response:")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)