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
from typing import Tuple, Union

from pydantic import BaseModel

from tests.functional.utils.inference.serving.common import LLMCommonWrapper
from tests.functional.utils.logger import get_logger

logger = get_logger(__name__)

OPENAI = "OPENAI"


class OpenAIWrapper(LLMCommonWrapper):
    API_KEY_UNUSED = "unused"
    CHAT_COMPLETIONS = "chat/completions"
    COMPLETIONS = "completions"
    RESPONSES = "responses"
    EMBEDDINGS = "embeddings"
    PREDICT = CHAT_COMPLETIONS
    IMAGES_GENERATIONS = "images/generations"
    IMAGES_EDITS = "images/edits"
    MODELS_LIST = "models"
    MODELS_RETRIEVE = "models/{model}"
    AUDIO_SPEECH = "audio/speech"
    AUDIO_TRANSCRIPTIONS = "audio/transcriptions"
    AUDIO_TRANSLATIONS = "audio/translations"
    AVAILABLE_COMPLETIONS_ENDPOINTS = [CHAT_COMPLETIONS, COMPLETIONS]
    AVAILABLE_TEXT_GENERATION_ENDPOINTS = [CHAT_COMPLETIONS, COMPLETIONS, RESPONSES]
    AVAILABLE_IMAGES_ENDPOINTS = [IMAGES_GENERATIONS, IMAGES_EDITS]
    AVAILABLE_MODELS_ENDPOINTS = [MODELS_LIST, MODELS_RETRIEVE]
    AVAILABLE_AUDIO_ENDPOINTS = [AUDIO_SPEECH, AUDIO_TRANSCRIPTIONS, AUDIO_TRANSLATIONS]
    AVAILABLE_AUDIO_TTS_ENDPOINTS = [AUDIO_SPEECH]
    AVAILABLE_AUDIO_ASR_ENDPOINTS = [AUDIO_TRANSCRIPTIONS, AUDIO_TRANSLATIONS]

    @staticmethod
    def prepare_body_dict(input_objects: dict, request_format=None, **kwargs):
        """
        example body dict for chat/completions:
            {
                "model": "facebook/opt-125m",
                "stream": false,
                "max_tokens": 30,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"}
                ]
            }
        example body dict for completions:
            {
                "model": "facebook/opt-125m",
                "stream": false,
                "max_tokens": 30,
                "prompt": "Who won the world series in 2020?"
            }
        """

        model = kwargs.get("model", None)
        assert model is not None, "No model provided"
        model_name = model.name

        endpoint = kwargs.get("endpoint", OpenAIWrapper.CHAT_COMPLETIONS)
        if endpoint == OpenAIWrapper.CHAT_COMPLETIONS:
            body_dict = {
                "model": model_name,
                "messages": ChatCompletionsApi.prepare_chat_completions_input_content(input_objects),
            }
        elif endpoint == OpenAIWrapper.COMPLETIONS:
            body_dict = {
                "model": model_name,
                "prompt": CompletionsApi.prepare_completions_input_content(input_objects),
            }
        elif endpoint == OpenAIWrapper.EMBEDDINGS:
            body_dict = {
                "model": model_name,
                "input": EmbeddingsApi.prepare_embeddings_input_content(input_objects),
            }
        elif endpoint == OpenAIWrapper.IMAGES_EDITS:
            prompt, image_path = ImagesApi.prepare_image_edit_input_content(input_objects)
            body_dict = {
                "model": model_name,
                "image": image_path,
                "prompt": prompt,
            }
        elif endpoint == OpenAIWrapper.IMAGES_GENERATIONS:
            body_dict = {
                "model": model_name,
                "prompt": ImagesApi.prepare_image_generation_input_content(input_objects),
            }
        elif endpoint == OpenAIWrapper.RESPONSES:
            body_dict = {
                "model": model_name,
                "input": ResponsesApi.prepare_responses_input_content(input_objects),
            }
        elif endpoint == OpenAIWrapper.AUDIO_SPEECH:
            reference_text, _ = AudioApi.prepare_audio_input_content(input_objects)
            body_dict = {
                "model": model_name,
                "input": reference_text,
            }
        elif endpoint in OpenAIWrapper.AVAILABLE_AUDIO_ASR_ENDPOINTS:
            _, reference_audio_file = AudioApi.prepare_audio_input_content(input_objects)
            body_dict = {
                "model": model_name,
                "file": reference_audio_file,
            }
        else:
            raise NotImplementedError(f"Invalid endpoint: {endpoint}")
        return LLMCommonWrapper.prepare_body_dict_from_request_params(OpenAIRequestParams, body_dict, **kwargs)


@dataclass
class OpenAIRequestParamsNames:
    STREAM = "stream"
    MAX_TOKENS = "max_tokens"
    IGNORE_EOS = "ignore_eos"
    N = "n"
    BEST_OF = "best_of"
    LENGTH_PENALTY = "length_penalty"
    TOP_P = "top_p"
    TOP_K = "top_k"
    REPETITION_PENALTY = "repetition_penalty"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    SEED = "seed"


@dataclass
class ChatCompletionsApi:
    ROLE = "role"
    ROLE_USER = "user"
    ROLE_SYSTEM = "system"
    ROLE_ASSISTANT = "assistant"
    ROLE_TOOL = "tool"
    CONTENT = "content"
    CONTENT_TYPE = "type"
    CONTENT_TYPE_TEXT = "text"
    CONTENT_TYPE_IMAGE_URL = "image_url"
    TOOL_CALLS = "tool_calls"
    TOOL_CALL_ID = "tool_call_id"
    REASONING_CONTENT = "reasoning_content"

    @classmethod
    def prepare_chat_completions_input_content(cls, input_objects):
        messages = []
        for input_name in input_objects:
            for objects in input_objects[input_name]:
                if len(objects) == 2:
                    role, content = objects
                    messages.append({cls.ROLE: role, cls.CONTENT: content})
                else:
                    role = objects[0]
                    if role == ChatCompletionsApi.ROLE_ASSISTANT:
                        messages.append({
                            cls.ROLE: role,
                            cls.CONTENT: objects[1],
                            cls.TOOL_CALLS: objects[2],
                            cls.REASONING_CONTENT: objects[3],
                        })
                    elif role == ChatCompletionsApi.ROLE_TOOL:
                        messages.append({
                            cls.ROLE: role,
                            cls.CONTENT: objects[1],
                            cls.TOOL_CALL_ID: objects[2],
                        })
                    else:
                        raise NotImplementedError()
        return messages


class CompletionsApi:

    @staticmethod
    def prepare_completions_input_content(input_objects):
        for input_name in input_objects:
            prompt = " ".join([content for _, content in input_objects[input_name]])
            return prompt


class ImagesApi:
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"

    @staticmethod
    def prepare_image_generation_input_content(input_objects):
        for input_name in input_objects:
            prompt = " ".join(input_objects[input_name])
            return prompt

    @staticmethod
    def prepare_image_edit_input_content(input_objects):
        for input_value in input_objects.values():
            prompt = input_value[0]
            image_path = input_value[1]
            return prompt, image_path


class ResponsesApi:
    CONTENT_TYPE_TEXT = "input_text"
    CONTENT_TYPE_IMAGE_URL = "input_image"

    # Mapping from ChatCompletionsApi content types to ResponsesApi content types
    _CONTENT_TYPE_MAP = {
        ChatCompletionsApi.CONTENT_TYPE_TEXT: CONTENT_TYPE_TEXT,
        ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: CONTENT_TYPE_IMAGE_URL,
    }

    @classmethod
    def _convert_content(cls, content):
        if not isinstance(content, list):
            return content
        converted_content = []
        for content_item in content:
            if isinstance(content_item, dict) and ChatCompletionsApi.CONTENT_TYPE in content_item:
                original_type = content_item[ChatCompletionsApi.CONTENT_TYPE]
                if original_type in cls._CONTENT_TYPE_MAP:
                    content_item[ChatCompletionsApi.CONTENT_TYPE] = cls._CONTENT_TYPE_MAP[original_type]
                converted_content.append(content_item)
            else:
                converted_content.append(content_item)
        return converted_content

    @classmethod
    def prepare_responses_input_content(cls, input_objects):
        input_content = []
        for input_name in input_objects:
            for objects in input_objects[input_name]:
                role, content = objects[0], objects[1]
                content = cls._convert_content(content)     # update datasets for VLMs
                input_content.append({"role": role, "content": content})
        return input_content


class EmbeddingsApi:

    @staticmethod
    def prepare_embeddings_input_content(input_objects):
        for input_name in input_objects:
            return input_objects[input_name]


class AudioApi:

    @staticmethod
    def prepare_audio_input_content(input_objects) -> Tuple[str, str]:
        for input_name in input_objects:
            return input_objects[input_name][0]


class OpenAIRequestParams:

    def prepare_dict(self, set_null_values=False, **kwargs):
        if set_null_values:
            request_params_dict = {key: value for key, value in vars(self).items()}
        else:
            request_params_dict = {key: value for key, value in vars(self).items() if value is not None}
        return request_params_dict


@dataclass
class OpenAICommonCompletionsRequestParams(OpenAIRequestParams):
    stream: bool = None
    stream_options: dict = None
    max_tokens: int = None
    n: int = None
    temperature: float = None
    top_p: float = None
    frequency_penalty: float = None
    presence_penalty: float = None
    seed: int = None
    stop: Union[str, list] = None

    def set_default_values(self, **kwargs):
        self.stream = kwargs.get("stream", False)
        self.n = 1
        self.temperature = 1.0
        self.top_p = 1.0
        self.seed = 0
        self.stop = ","


@dataclass
class OpenAIChatCompletionsRequestParams(OpenAICommonCompletionsRequestParams):
    logprobs: bool = None
    response_format: dict = None
    tools: list = None
    tool_choice: Union[dict, str] = None
    response_format: BaseModel = None

    def set_default_values(self, **kwargs):
        super().set_default_values(**kwargs)
        if not self.stream:
            self.logprobs = False


@dataclass
class OpenAICompletionsRequestParams(OpenAICommonCompletionsRequestParams):
    best_of: int = None
    logprobs: int = None

    def set_default_values(self, **kwargs):
        super().set_default_values(**kwargs)
        if not self.stream:
            self.best_of = 1
            self.logprobs = 1


@dataclass
class OpenAICommonImagesRequestParams(OpenAIRequestParams):
    size: str = None
    n: int = None
    response_format: str = None

    def set_default_values(self, **kwargs):
        self.size = "512x512"
        self.n = 1
        self.response_format = "b64_json"


@dataclass
class OpenAIImagesGenerationsRequestParams(OpenAICommonImagesRequestParams):
    pass


@dataclass
class OpenAIImagesEditsRequestParams(OpenAICommonImagesRequestParams):
    pass


class OpenAIFinishReason:
    LENGTH = "length"
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    MAX_TOKENS = "max_tokens"


@dataclass
class OpenAIResponsesRequestParams(OpenAIRequestParams):
    stream: bool = None
    max_output_tokens: int = None
    temperature: float = None
    top_p: float = None
    tools: list = None
    tool_choice: Union[dict, str] = None

    def set_default_values(self, **kwargs):
        self.stream = kwargs.get("stream", False)
        self.temperature = 1.0
        self.top_p = 1.0


@dataclass
class OpenAIEmbeddingsRequestParams(OpenAIRequestParams):
    encoding_format: str = None

    def set_default_values(self):
        self.encoding_format = "float"


@dataclass
class OpenAIAudioSpeechRequestParams(OpenAIRequestParams):

    def set_default_values(self, **kwargs):
        # No defaults needed for TTS; kept for interface compatibility
        pass


@dataclass
class OpenAIAudioTranscriptionsRequestParams(OpenAIRequestParams):
    language: str = None
    temperature: float = None
    timestamp_granularities: list = None

    def set_default_values(self, **kwargs):
        self.language = "en"
        self.temperature = 0.0
        self.timestamp_granularities = ["segment"]


@dataclass
class OpenAIAudioTranslationsRequestParams(OpenAIRequestParams):
    temperature: float = None

    def set_default_values(self, **kwargs):
        self.temperature = 0.0
