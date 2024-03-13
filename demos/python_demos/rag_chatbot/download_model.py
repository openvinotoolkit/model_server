#*****************************************************************************
# Copyright 2024 Intel Corporation
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
#*****************************************************************************
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from servable_stream.config import SUPPORTED_LLM_MODELS

import argparse

parser = argparse.ArgumentParser(description='Script to download LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot')

supported_models_list = [model_name for model in SUPPORTED_LLM_MODELS.values() for model_name in model]

parser.add_argument('--model',
                    required=True,
                    choices=supported_models_list,
                    help='Select the LLM model out of supported list')
args = vars(parser.parse_args())

SELECTED_MODEL = args['model']
LANGUAGE = 'English'

model_configuration = SUPPORTED_LLM_MODELS[LANGUAGE][SELECTED_MODEL]

model_id = model_configuration["model_id"]

MODEL_PATH = "./" + SELECTED_MODEL

print('Downloading and converting...')
ov_model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    device='CPU',
    compile=False,
    trust_remote_code=True,
    load_in_8bit=False)

print(f'Saving model to {MODEL_PATH} ...')
ov_model.save_pretrained(MODEL_PATH)
print('Done.')

print(f'Downloading tokenizer to {MODEL_PATH} ...')
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f'Saving tokenizer to {MODEL_PATH} ...')
tok.save_pretrained(MODEL_PATH)
print('Done.')

if not ov_model.stateful:
    print("WARNING: Saved model is not stateful")
    exit(1)
