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
#

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


models = [
    "facebook/opt-125m",
]

def save_ov_model_from_optimum(model, hf_tokenizer, model_path: str):
    model.save_pretrained(model_path)
    # convert tokenizers as well
    from openvino_tokenizers import convert_tokenizer
    from openvino import serialize
    tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True, skip_special_tokens=True)
    serialize(tokenizer, f"{model_path}/openvino_tokenizer.xml")
    serialize(detokenizer, f"{model_path}/openvino_detokenizer.xml")

def get_model_and_tokenizer(model_id: str, use_optimum = True):
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True) if use_optimum else \
            AutoModelForCausalLM.from_pretrained(model_id)
    return model, hf_tokenizer

def prepare_models(model_id: str, tmp_path: str):
    use_optimum = True
    model_path : str = f"{tmp_path}/{model_id}"
    model, hf_tokenizer = get_model_and_tokenizer(model_id, use_optimum)

    if use_optimum:
        save_ov_model_from_optimum(model, hf_tokenizer, model_path)


def main():
    import os
    for model_id in models:
        prepare_models(model_id, os.path.dirname(__file__))

if __name__ == "__main__":
    main()