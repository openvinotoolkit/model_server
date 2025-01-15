#
# Copyright (c) 2025 Intel Corporation
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


from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
device = "cpu"
# Make prompts
prompt = [
'''"[user] Write a SQL query to answer the question based on the table schema.\n
\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n
\n question: Name the ICAO for lilongwe international airport [/user] [assistant]''']

# Load Models
base_model = "meta-llama/Llama-2-7b-hf" 
peft_adapter = "yard1/llama-2-7b-sql-lora-test"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model)


def generate_base(model, prompt, tokenizer):    
    print("Generating results")
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    res = model.generate(**tokens, max_new_tokens=100)
    res_sentences = [tokenizer.decode(i) for i in res]
    print("Results:",res_sentences)

def merge_models(model, adapter):
    print("Merging model with adapter")
    adapter_tokenizer = AutoTokenizer.from_pretrained(adapter)
    model.resize_token_embeddings(len(adapter_tokenizer), mean_resizing=False)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.eval()
    model = model.to(device)
    return model, adapter_tokenizer

print("BASE MODEL")
generate_base(model, prompt, tokenizer)
model, adapter_tokenizer = merge_models(model, peft_adapter)
print("MERGED MODEL")
generate_base(model, prompt, adapter_tokenizer)



