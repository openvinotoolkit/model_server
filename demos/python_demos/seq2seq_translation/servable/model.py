#*****************************************************************************
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
#*****************************************************************************

from pyovms import Tensor
from optimum.intel import OVModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline
import time

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        model_id = "echarlaix/t5-small-openvino"
        model = OVModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
        return True

    def execute(self, inputs: list):
        text = bytes(inputs[0]).decode()
        results = self.pipe(text)
        translation = results[0]["translation_text"]
        return [Tensor("OUTPUT", translation.encode())]

