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
from optimum.intel.openvino import OVModelForCausalLM

model_name = 'togethercomputer/RedPajama-INCITE-Chat-3B-v1'
#model_name = 'meta-llama/Llama-2-7b-chat-hf'  # Edit for other models, for example llama
MODEL_PATH = './model'
OV_CONFIG = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}

print('Downloading and converting...')
ov_model = OVModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    device='CPU',
    compile=False,
    ov_config=OV_CONFIG)

print(f'Saving to {MODEL_PATH} ...')
ov_model.save_pretrained(MODEL_PATH)
print('Done.')
