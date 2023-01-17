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
import os
import time
import ovmsclient
import torch
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Demo for GPT-J causal LM requests using ovmsclient gRPC API')

parser.add_argument('--input', required=True, help='Beginning of a sentence', type=str)
parser.add_argument('--eos_token_id', required=False, help='End of sentence token', type=int, default=198)
args = vars(parser.parse_args())

client = ovmsclient.make_grpc_client("localhost:11340")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

input_sentence = args['input']
eos_token_id = args['eos_token_id']

iteration = 0
while True:
    inputs = tokenizer(input_sentence, return_tensors="np")
    start_time = time.time()
    results = client.predict(inputs=dict(inputs), model_name='resnet')
    last_latency = time.time() - start_time
    predicted_token_id = token = torch.argmax(torch.nn.functional.softmax(torch.Tensor(results[0,-1,:]),dim=-1),dim=-1)
    word = tokenizer.decode(predicted_token_id)
    input_sentence += word
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Iteration: {iteration}\nLast predicted token: {predicted_token_id}\nLast latency: {last_latency}s\n{input_sentence}")
    iteration += 1
    if predicted_token_id == eos_token_id:
        break

