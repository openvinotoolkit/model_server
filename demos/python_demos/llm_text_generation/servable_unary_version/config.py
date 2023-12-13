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

# This entire file is for customization dependin on the LLM model.
# For now only red pajama is supported.
# More info: https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/config.py

# TODO: This is useful for other models,
# which DO require prompting.
# Red pajama does not need it. Llama does.
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


def default_partial_text_processor(partial_text: str, new_text: str):
    partial_text += new_text
    return partial_text


def red_pijama_partial_text_processor(partial_text, new_text):
    if new_text == "<":
        return partial_text

    partial_text += new_text
    return partial_text.split("<bot>:")[-1]


def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text


SUPPORTED_MODELS = {
    "red-pajama-3b-chat": {
        "model_id": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        #"remote": False,
        "start_message": "",
        "history_template": "\n<human>:{user}\n<bot>:{assistant}",
        "stop_tokens": [29, 0],
        "partial_text_processor": red_pijama_partial_text_processor,
        "current_message_template": "\n<human>:{user}\n<bot>:{assistant}",
    },
    "llama-2-chat-7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        #"remote": False,
        "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
        "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
        "current_message_template": "{user} [/INST]{assistant}",
        "tokenizer_kwargs": {"add_special_tokens": False},
        "partial_text_processor": llama_partial_text_processor,
        #"revision": "5514c85fedd6c4fc0fc66fa533bc157de520da73",
    }
}
