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
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

import threading

from config import SUPPORTED_MODELS

SELECTED_MODEL = 'red-pajama-3b-chat'
model_configuration = SUPPORTED_MODELS[SELECTED_MODEL]

MODEL_PATH = "/model"  # relative to container
OV_CONFIG = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}

# Model specific configuration
model_name = model_configuration["model_id"]
history_template = model_configuration["history_template"]
current_message_template = model_configuration["current_message_template"]
start_message = model_configuration["start_message"]
stop_tokens = model_configuration.get("stop_tokens")
tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
text_processor = model_configuration.get("partial_text_processor")


# HF class that is capable of stopping the generation
# when given tokens appear in specific order
class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


# For multi Q&A use cases
# Taken from notebook:
# https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb
def convert_history_to_text(history):
    """
    function for conversion history stored as list pairs of user and assistant messages to string according to model expected conversation template
    Params:
      history: dialogue history
    Returns:
      history in text format
    """
    text = start_message + "".join(
        [
            "".join(
                [history_template.format(num=round, user=item[0], assistant=item[1])]
            )
            for round, item in enumerate(history[:-1])
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    current_message_template.format(
                        num=len(history) + 1,
                        user=history[-1][0],
                        assistant=history[-1][1],
                    )
                ]
            )
        ]
    )
    return text


class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        print("-------- Running initialize")
        # model_id = "echarlaix/t5-small-openvino"
        # model = OVModelForSeq2SeqLM.from_pretrained(model_id)
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
        self.ov_model = OVModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device="CPU",
            ov_config=OV_CONFIG,
            config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True))
        print("-------- Model loaded, loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("-------- Tokenizer loaded")
        return True

    def execute(self, inputs: list):
        print("-------- Running execute")
        text = bytes(inputs[0]).decode()
        temporal_history = [[text, ""]]

        messages = convert_history_to_text(temporal_history)
        print('------------------- ', messages)
        input_ids = self.tokenizer(messages, return_tensors="pt", **tokenizer_kwargs).input_ids
        streamer = TextIteratorStreamer(self.tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=1024,
            temperature=1.0,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
        )
        if stop_tokens is not None:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopOnTokens(stop_tokens)  # TODO: Stop tokens might be string as well
            ])
        
        def generate():
            self.ov_model.generate(**generate_kwargs)

        t1 = threading.Thread(target=generate)
        t1.start()

        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            print('----', partial_text)
        

        return [Tensor("OUTPUT", partial_text.encode())]
        #return [Tensor("OUTPUT", b'asd')]
        # text = bytes(inputs[0]).decode()
        # print(text)
        # results = self.pipe(text)
        # translation = results[0]["translation_text"]
        # print(translation)
        # return [Tensor("OUTPUT", translation.encode())]

