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
import argparse

from optimum.intel.openvino import OVModelForCausalLM
from huggingface_hub import login, whoami

parser = argparse.ArgumentParser(description='Download script for llama-7b-chat')

parser.add_argument('--token', required=False, help='Huggingfaces login token')
args = parser.parse_args()

try:
    whoami()
    print('Authorization token already provided')
except OSError:
    if args.token is None:
        print("need to provide HF authorization --token param")
        exit(1)
    login(args.token)

print('Downloading and converting...')
ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}
ov_model = OVModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    export=True,
    device='CPU',
    compile=False,
    ov_config=ov_config)

print('Saving to ./models ...')
ov_model.save_pretrained('./models/llama-2-7b-hf/1/')
print('Done.')
