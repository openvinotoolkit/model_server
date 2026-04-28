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

from dataclasses import dataclass
from typing import Union

from tests.functional.utils.inference.serving.openai import (
    OpenAIChatCompletionsRequestParams,
    OpenAICommonCompletionsRequestParams,
    OpenAICommonImagesRequestParams,
    OpenAICompletionsRequestParams,
    OpenAIEmbeddingsRequestParams,
    OpenAIImagesEditsRequestParams,
    OpenAIImagesGenerationsRequestParams,
    OpenAIResponsesRequestParams,
    OpenAIRequestParams,
    OpenAIWrapper,
    OpenAIAudioTranscriptionsRequestParams,
    OpenAIAudioTranslationsRequestParams,
    OpenAIAudioSpeechRequestParams,
)
from tests.functional.constants.ovms import Ovms


@dataclass
class MaxTokensValues:
    DEFAULT = 30
    DEFAULT_COMPARE_JINJA = 50
    VLM_COMPARE = 50
    COMPARE = 100
    COMPARE_LONG = 1000
    CACHE_OVERLOAD = 100000
    SHORT = 10
    LONG = 10000
    TOOLS = 2048


class TemperatureValues:
    TEST_DEFAULT = 0
    OVMS_DEFAULT = 1


class EncodingFormatValues:
    FLOAT = "float"
    BASE64 = "base64"

    @classmethod
    def values(cls):
        values = [
            value for key, value in vars(cls).items() if not key.startswith("__") and not isinstance(value, classmethod)
        ]
        return values


class MaxPromptLenValues:
    NPU_DEFAULT = 10240


class ImagesRequestParamsValues:
    N_DEFAULT = 1
    NUM_INFERENCE_STEPS_DEFAULT = 50
    NUM_INFERENCE_STEPS_FLUX = 3
    NUM_INFERENCE_STEPS_SDXL = 25
    NUM_INFERENCE_STEPS_DREAMLIKE_INPAINTING = 100
    NUM_INFERENCE_STEPS_NPU = 1
    RNG_SEED_DEFAULT = 42
    SIZE_DEFAULT = "512x512"
    SIZE_EDITS_DEFAULT = "336x224"
    STRENGTH_DEFAULT = 0.7
    MIXED_NPU_DEVICE = "NPU NPU GPU"


class ResponseFormatValues:
    DEFAULT = {
        "type": "json_schema",
        "json_schema": {
            "description": "city and country sch",
            "schema": {
                "properties": {
                   "city": {
                      "title": "City",
                      "type": "string",
                   },
                   "country": {
                      "title": "Country",
                      "type": "string",
                   },
                },
                "required": ["city", "country"],
                "additionalProperties": False,
            },
            "name": "schema_name",
            "strict": False,
        },
    }


@dataclass
class OvmsCommonRequestParams:

    def prepare_dict_with_extra_body(self, openai_base_classes: list):
        request_params_dict = {"extra_body": {}}
        for key, value in vars(self).items():
            if value is not None:
                if any(key in vars(openai_base_class) for openai_base_class in openai_base_classes):
                    request_params_dict[key] = value
                else:
                    request_params_dict["extra_body"][key] = value
        return request_params_dict


@dataclass
class OvmsCommonCompletionsRequestParams(OpenAICommonCompletionsRequestParams, OvmsCommonRequestParams):
    ignore_eos: bool = None
    length_penalty: float = None
    include_stop_str_in_output: bool = None
    top_k: int = None
    repetition_penalty: float = None
    num_assistant_tokens: int = None
    assistant_confidence_threshold: float = None
    max_ngram_size: int = None

    def set_default_values(self, **kwargs):
        super().set_default_values(**kwargs)
        self.ignore_eos = False
        self.length_penalty = 1.0
        self.repetition_penalty = 1.0


@dataclass
class OvmsChatCompletionsRequestParams(OvmsCommonCompletionsRequestParams, OpenAIChatCompletionsRequestParams):
    # Some request parameters are supported by OVMS but not by OpenAI API. For full list go to:
    # https://github.com/openvinotoolkit/model_server/blob/main/docs/model_server_rest_api_chat.md
    best_of: int = None
    tools: list = None
    tool_choice: Union[dict, str] = None
    chat_template_kwargs: dict = None

    def prepare_dict(self, set_null_values=False, use_extra_body=True):
        if use_extra_body:
            return self.prepare_dict_with_extra_body(
                [OpenAICommonCompletionsRequestParams, OpenAIChatCompletionsRequestParams],
            )
        else:
            return super().prepare_dict(set_null_values=set_null_values)

    def set_default_values(self, **kwargs):
        super().set_default_values(**kwargs)
        if not self.stream:
            self.best_of = 1


@dataclass
class OvmsCompletionsRequestParams(OvmsCommonCompletionsRequestParams, OpenAICompletionsRequestParams):
    # Some of the request parameters are supported by OVMS but not by OpenAI API. For full list go to:
    # https://github.com/openvinotoolkit/model_server/blob/main/docs/model_server_rest_api_completions.md

    def prepare_dict(self, set_null_values=False, use_extra_body=True):
        if use_extra_body:
            return self.prepare_dict_with_extra_body(
                [OpenAICommonCompletionsRequestParams, OpenAICompletionsRequestParams],
            )
        else:
            return super().prepare_dict(set_null_values=set_null_values)


@dataclass
class OvmsResponsesRequestParams(OpenAIResponsesRequestParams, OvmsCommonRequestParams):
    ignore_eos: bool = None
    stop: Union[str, list] = None
    top_k: int = None
    include_stop_str_in_output: bool = None
    repetition_penalty: float = None
    frequency_penalty: float = None
    presence_penalty: float = None
    seed: int = None
    best_of: int = None
    length_penalty: float = None
    n: int = None
    num_assistant_tokens: int = None
    assistant_confidence_threshold: float = None
    max_ngram_size: int = None

    def set_default_values(self, **kwargs):
        super().set_default_values(**kwargs)
        self.ignore_eos = False
        self.stop = ","

    def prepare_dict(self, set_null_values=False, use_extra_body=True):
        if use_extra_body:
            return self.prepare_dict_with_extra_body([OpenAIResponsesRequestParams])
        else:
            return super().prepare_dict(set_null_values=set_null_values)


@dataclass
class OvmsCommonImagesRequestParams(OpenAICommonImagesRequestParams, OvmsCommonRequestParams):
    prompt_2: str = None
    prompt_3: str = None
    negative_prompt: str = None
    negative_prompt_2: str = None
    negative_prompt_3: str = None
    num_images_per_prompt: int = None
    num_inference_steps: int = None
    guidance_scale: float = None
    rng_seed: int = None
    max_sequence_length: int = None
    height: int = None
    width: int = None

    def set_default_values(self, **kwargs):
        super().set_default_values(**kwargs)
        self.num_inference_steps = 50
        self.rng_seed = 42


@dataclass
class OvmsImagesGenerationsRequestParams(OvmsCommonImagesRequestParams, OpenAIImagesGenerationsRequestParams):

    def prepare_dict(self, set_null_values=False, use_extra_body=True):
        if use_extra_body:
            return self.prepare_dict_with_extra_body(
                [OpenAICommonImagesRequestParams, OpenAIImagesGenerationsRequestParams],
            )
        else:
            return super().prepare_dict(set_null_values=set_null_values)


@dataclass
class OvmsImagesEditsRequestParams(OvmsCommonImagesRequestParams, OpenAIImagesEditsRequestParams):
    strength: float = None

    def prepare_dict(self, set_null_values=False, use_extra_body=True):
        if use_extra_body:
            return self.prepare_dict_with_extra_body(
                [OpenAICommonImagesRequestParams, OpenAIImagesEditsRequestParams],
            )
        else:
            return super().prepare_dict(set_null_values=set_null_values)

    def set_default_values(self, **kwargs):
        super().set_default_values(**kwargs)
        self.strength = 0.7


@dataclass
class OvmsEmbeddingsRequestParams(OpenAIEmbeddingsRequestParams):
    pass


@dataclass
class OvmsAudioTranscriptionsRequestParams(OpenAIAudioTranscriptionsRequestParams):
    pass


@dataclass
class OvmsAudioTranslationsRequestParams(OpenAIAudioTranslationsRequestParams):
    pass


@dataclass
class OvmsAudioSpeechRequestParams(OpenAIAudioSpeechRequestParams):
    pass


class OvmsOpenAIRequestParamsBuilder:

    def __init__(self, endpoint, **kwargs):
        self.endpoint = endpoint
        if self.endpoint == OpenAIWrapper.CHAT_COMPLETIONS:
            self.request_params = OvmsChatCompletionsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.COMPLETIONS:
            self.request_params = OvmsCompletionsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.RESPONSES:
            # translate chat/completions parameters
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens", None)
            self.request_params = OvmsResponsesRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.EMBEDDINGS:
            self.request_params = OvmsEmbeddingsRequestParams()
        elif self.endpoint == OpenAIWrapper.IMAGES_GENERATIONS:
            self.request_params = OvmsImagesGenerationsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.IMAGES_EDITS:
            self.request_params = OvmsImagesEditsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.MODELS_LIST:
            self.request_params = OpenAIRequestParams()     # no request parameters available
        elif self.endpoint == OpenAIWrapper.MODELS_RETRIEVE:
            self.request_params = OpenAIRequestParams()     # no request parameters available
        elif self.endpoint == OpenAIWrapper.AUDIO_TRANSCRIPTIONS:
            self.request_params = OvmsAudioTranscriptionsRequestParams()
        elif self.endpoint == OpenAIWrapper.AUDIO_TRANSLATIONS:
            self.request_params = OvmsAudioTranslationsRequestParams()
        elif self.endpoint == OpenAIWrapper.AUDIO_SPEECH:
            self.request_params = OvmsAudioSpeechRequestParams()
        else:
            raise NotImplementedError


@dataclass
class FuzzOvmsCommonCompletionsRequestParams(OvmsCommonCompletionsRequestParams):
    frequency_penalty: float = None
    logit_bias: dict = None
    logprobs: bool = None
    presence_penalty: float = None
    stop = None
    user: str = None
    top_k: int = None

    def set_default_values(self, **kwargs):
        self.stream = kwargs.get("stream", False)
        self.max_tokens = MaxTokensValues.DEFAULT
        self.temperature = 0
        if kwargs.get("enable_generic", False):
            self.stop = ","
            if self.stream:
                self.stream_options = {"include_usage": True}
                self.include_stop_str_in_output = True
            else:
                self.include_stop_str_in_output = False
                self.logprobs = False
            self.ignore_eos = False
        if kwargs.get("enable_beam_search", False):
            self.temperature = None
            self.n = 1
            if not self.stream:
                self.best_of = 1
            self.length_penalty = 1.0
        if kwargs.get("enable_multinomial", False):
            self.temperature = 1.0
            self.top_p = 1.0
            self.top_k = 1000
            self.repetition_penalty = 1.0
            self.frequency_penalty = 0.0
            self.presence_penalty = 0.0
            self.seed = 0


@dataclass
class FuzzOvmsChatCompletionsRequestParams(OvmsChatCompletionsRequestParams, FuzzOvmsCommonCompletionsRequestParams):

    def set_default_values(self, **kwargs):
        super(OvmsChatCompletionsRequestParams, self).set_default_values(**kwargs)
        if kwargs.get("enable_response_format", False):
            self.response_format = ResponseFormatValues.DEFAULT
        if kwargs.get("enable_tools", False):
            self.tools = Ovms.GET_WEATHER_TOOLS
            self.tool_choice = kwargs.get("tool_choice", Ovms.GET_WEATHER_TOOL_CHOICE)
            enable_thinking = kwargs.get("enable_thinking", False)
            self.chat_template_kwargs = {"enable_thinking": enable_thinking}


@dataclass
class FuzzOvmsCompletionsRequestParams(OvmsCompletionsRequestParams, FuzzOvmsCommonCompletionsRequestParams):
    echo: bool = None

    def set_default_values(self, **kwargs):
        super(OvmsCompletionsRequestParams, self).set_default_values(**kwargs)
        self.echo = False


@dataclass
class FuzzOvmsEmbeddingsRequestParams(OvmsEmbeddingsRequestParams):
    pass


class FuzzOvmsImagesEditsRequestParams(OvmsImagesEditsRequestParams):
    pass


class FuzzOvmsImagesEditsGenerationsRequestParams(OvmsImagesGenerationsRequestParams):
    pass


class FuzzOvmsAudioSpeechRequestParams(OvmsAudioSpeechRequestParams):
    pass


class FuzzOvmsAudioTranscriptionsRequestParams(OvmsAudioTranscriptionsRequestParams):
    pass


class FuzzOvmsAudioTranslationsRequestParams(OvmsAudioTranslationsRequestParams):
    pass


class FuzzOvmsOpenAIRequestParamsBuilder:
    # for request parameters that are not supported by OVMS but are supported by OpenAI

    def __init__(self, endpoint=None, **kwargs):
        self.endpoint = endpoint
        if self.endpoint == OpenAIWrapper.CHAT_COMPLETIONS:
            self.request_params = FuzzOvmsChatCompletionsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.COMPLETIONS:
            self.request_params = FuzzOvmsCompletionsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.EMBEDDINGS:
            self.request_params = FuzzOvmsEmbeddingsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.IMAGES_EDITS:
            self.request_params = FuzzOvmsImagesEditsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.IMAGES_GENERATIONS:
            self.request_params = FuzzOvmsImagesEditsGenerationsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.AUDIO_SPEECH:
            self.request_params = FuzzOvmsAudioSpeechRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.AUDIO_TRANSCRIPTIONS:
            self.request_params = FuzzOvmsAudioTranscriptionsRequestParams(**kwargs)
        elif self.endpoint == OpenAIWrapper.AUDIO_TRANSLATIONS:
            self.request_params = FuzzOvmsAudioTranslationsRequestParams(**kwargs)
        else:
            raise NotImplementedError
