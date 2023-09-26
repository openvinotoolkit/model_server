#
# Copyright (c) 2023 Intel Corporation
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
import numpy as np
import argparse
import os

from ovmsclient import make_grpc_client
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Inference script for generating text with llama')

parser.add_argument('--url', required=True, type=str, help='Specify url to grpc service')
parser.add_argument('--question', required=True, type=str, help='Question to selected actor')
parser.add_argument('--seed', required=False, type=int, default=0,
    help='Seed for next token selection algorithm. Providing different numbers will produce slightly different results.')
parser.add_argument('--actor', required=False, type=str, choices=['general-knowledge', 'python-programmer'], default='general-knowledge',
    help='Domain in which you want to interact with the model. Selects predefined pre-prompt.')
args = parser.parse_args()

np.random.seed(args.seed)
client = make_grpc_client(args.url)

GENERAL_PRE_PROMPT = """You are super-human who is able to answer any question. All questions and answers should end with [EOS].
Question: What is the capital of Poland? [EOS]
Answer: Warsaw. [EOS]

Question: <Q> [EOS]
Answer:"""

PROGRAMMER_PRE_PROMPT = """You are super-programmer who is brilliant python coder. All the answers contain code snippets ending with [EOS].
Question: Write short python function to sum numbers in a list. [EOS]
Answer:
def sum_all_numbers(numbers):
  result = 0
    for num in numbers:
        result += num
    return result [EOS]

Question: <Q> [EOS]
Answer:"""

if args.actor == 'general-knowledge':
    PRE_PROMPT = GENERAL_PRE_PROMPT
else:
    PRE_PROMPT = PROGRAMMER_PRE_PROMPT

PRE_PROMPT = PRE_PROMPT.replace("<Q>", args.question)


def prepare_preprompt_kv_cache(preprompt):
    res = tokenizer(preprompt, return_tensors="np", add_special_tokens=False)
    inputs = dict(
        input_ids = res.input_ids,                  # [1,X]
        attention_mask = res.attention_mask         # [1,X]
    )
    for i in range(32):
        inputs[f"past_key_values.{i}.key"] = np.zeros((1, 32, 0, 128), dtype=np.float32)
        inputs[f"past_key_values.{i}.value"] = np.zeros((1, 32, 0, 128), dtype=np.float32)
    return client.predict(inputs=inputs, model_name='llama')


def generate_next_inputs(previous_result, number_of_previous_tokens):
    probs = previous_result['logits'][0, -1, :]             # 1,N,32000

    probs = np.exp(probs)/sum(np.exp(probs))                # softmax
    next_token = np.random.choice(len(probs), p=probs)


    next_inputs = dict(
        input_ids = np.array([next_token], dtype=np.int64).reshape((1,1)),
        attention_mask =np.ones((1, number_of_previous_tokens+1), dtype=np.int64)
    )
    for j in range(32):
        next_inputs[f"past_key_values.{j}.key"] = previous_result[f"present.{j}.key"]
        next_inputs[f"past_key_values.{j}.value"] = previous_result[f"present.{j}.value"]
    return next_inputs, next_token


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
    actual_token_list = list(tokenizer(PRE_PROMPT, return_tensors="np", add_special_tokens=False).input_ids[0])
    content_so_far = PRE_PROMPT

    results = prepare_preprompt_kv_cache(PRE_PROMPT)
    while True:
        next_inputs, token = generate_next_inputs(results, len(actual_token_list))
        results = client.predict(inputs=next_inputs, model_name='llama')

        actual_token_list.append(token)
        actual_content = tokenizer.decode(actual_token_list)
        
        print(actual_content.replace(content_so_far, ''), end='', flush=True)
        content_so_far = actual_content

        if actual_content.endswith("[EOS]"):
            break
