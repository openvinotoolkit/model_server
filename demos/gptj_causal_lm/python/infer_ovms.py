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
import ovmsclient
import torch
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Demo for simple inference to GPT-J-6b model')
parser.add_argument('--url', required=False, help='Url to connect to', default='localhost:9000')
parser.add_argument('--model_name', required=False, help='Model name in the serving', default='gpt-j-6b')
args = vars(parser.parse_args())

client = ovmsclient.make_grpc_client(args['url'])
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

input_sentence = "OpenVINO Model Server is"
inputs = tokenizer(input_sentence, return_tensors="np")
results = client.predict(inputs=dict(inputs), model_name=args['model_name'])

predicted_token_id = torch.argmax(torch.nn.functional.softmax(torch.Tensor(results[0,-1,:]),dim=-1),dim=-1)
word = tokenizer.decode(predicted_token_id)

print(results)
print('predicted word:', word)
