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
model = "meta-llama/Llama-2-7b-chat-hf"

print("unary completions")
stream = client.completions.create(
    model=model,
    prompt="hello",
    max_tokens=100,
    stream=False
)
print(stream)

print("stream completions")
stream = client.completions.create(
    model=model,
    prompt="hello",
    max_tokens=100,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="")

print("unary chat completions")
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=100,
    stream=False
)
print(response)

print("stream chat completions")
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=100,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")