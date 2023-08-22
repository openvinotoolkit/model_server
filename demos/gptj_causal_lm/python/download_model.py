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
import logging as log
from transformers import AutoTokenizer, AutoModelForCausalLM

log.basicConfig(level=log.DEBUG)

log.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
log.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
log.info("Saving the tokenizer to pytorch format...")
tokenizer.save_pretrained('local-pt-checkpoint')
log.info("Saving the model to pytorch format...")
model.save_pretrained('local-pt-checkpoint')
log.info("Done.")
