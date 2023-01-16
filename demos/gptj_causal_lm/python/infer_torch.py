#
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
#
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

input_sentence = "OpenVINO Model Server is"
inputs = tokenizer(input_sentence, return_tensors="pt")

outputs = model(**inputs, labels=inputs["input_ids"])
predicted_token_id = torch.argmax(torch.nn.Softmax(dim=-1)(outputs.logits[0,-1,:]))
word = tokenizer.decode(predicted_token_id)

print(outputs.logits)
print('predicted word:', word)
