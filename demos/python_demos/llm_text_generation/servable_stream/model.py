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

import os
import threading
import numpy as np
import torch

from typing import Optional, List, Tuple
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList, set_seed
from tritonclient.utils import deserialize_bytes_tensor, serialize_byte_tensor

from pyovms import Tensor

from config import SUPPORTED_LLM_MODELS, BatchTextIteratorStreamer

SELECTED_MODEL = os.environ.get('SELECTED_MODEL', 'tiny-llama-1b-chat')
LANGUAGE = os.environ.get("LANGUAGE", 'English')
SEED = os.environ.get("SEED")

print("SELECTED MODEL", SELECTED_MODEL, flush=True)
model_configuration = SUPPORTED_LLM_MODELS[LANGUAGE][SELECTED_MODEL]

MODEL_PATH = "/model"  # relative to container
OV_CONFIG = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by de

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text

text_processor = model_configuration.get(
    "partial_text_processor", default_partial_text_processor
)

# Model specific configuration
model_name = model_configuration["model_id"]
history_template = model_configuration["history_template"]
current_message_template = model_configuration["current_message_template"]
start_message = model_configuration["start_message"]
stop_tokens = model_configuration.get("stop_tokens")
tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # For models with tokenizer with uninitialized pad token

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


if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = tokenizer.convert_tokens_to_ids(stop_tokens)

    stop_tokens = [StopOnTokens(stop_tokens)]


# For multi Q&A use cases
# Taken from notebook:
# https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/
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


def deserialize_prompts(batch_size, input_tensor):
    np_arr = deserialize_bytes_tensor(bytes(input_tensor))
    return [arr.decode() for arr in np_arr]


def serialize_completions(batch_size, result):
    return [Tensor("completion", serialize_byte_tensor(
        np.array(result, dtype=np.object_)).item(), shape=[batch_size], datatype="BYTES")]


class OvmsPythonModel:
    def initialize(self, kwargs: dict):
        print("-------- Running initialize", flush=True)
        self.ov_model = OVModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device="AUTO",
            ov_config=OV_CONFIG,
            config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True))
        print("-------- Model loaded", flush=True)
        return True

    def execute(self, inputs: list):
        print(f"-------- Running execute, shape: {inputs[0].shape}", flush=True)
        batch_size = inputs[0].shape[0]
        prompts = deserialize_prompts(batch_size, inputs[0])
        messages = [convert_history_to_text([[prompt, ""]]) for prompt in prompts]
        tokens = tokenizer(messages, return_tensors="pt", **tokenizer_kwargs, padding=True)

        if batch_size == 1:
            streamer = TextIteratorStreamer(tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        else:
            streamer = BatchTextIteratorStreamer(batch_size=batch_size, tokenizer=tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            max_new_tokens=1024,
            temperature=1.0,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
        )
        if stop_tokens is not None:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

        ov_model_exec = self.ov_model.clone()
        token_count: List[int]= []
        def generate():
            result = ov_model_exec.generate(**tokens, **generate_kwargs)
            token_count.append(len([1 for x in result.numpy().flatten() if x not in tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)]))

        if SEED is not None: set_seed(int(SEED))
        t1 = threading.Thread(target=generate)
        t1.start()

        for partial_result in streamer:
            yield serialize_completions(batch_size, partial_result)
        t1.join()
        token_count[0] -= len(tokens["input_ids"].flatten())
        yield [Tensor("token_count", np.array(token_count, dtype=np.int32))]
        yield [Tensor("end_signal", "".encode())]
        print('end', flush=True)
