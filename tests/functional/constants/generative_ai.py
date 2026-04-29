#
# Copyright (c) 2026 Intel Corporation
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

class GenerativeAIFiles:
    CHAT_TEMPLATE_JINJA = "chat_template.jinja"
    TEMPLATE_JINJA = "template.jinja"
    TOKENIZER_CONFIG_JSON = "tokenizer_config.json"


class GenerativeAIPluginConfig:
    INFERENCE_PRECISION_HINT = "INFERENCE_PRECISION_HINT"
    KV_CACHE_PRECISION = "KV_CACHE_PRECISION"
    DEVICE_PROPERTIES = "DEVICE_PROPERTIES"
    NPUW_DEVICES = "NPUW_DEVICES"
    NPUW_ONLINE_PIPELINE = "NPUW_ONLINE_PIPELINE"
    MAX_PROMPT_LEN = "MAX_PROMPT_LEN"
    PROMPT_LOOKUP = "prompt_lookup"
