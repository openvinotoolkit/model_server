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

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

from openai import NotFoundError

from tests.functional.config import (
    logging_level_ovms,
    kv_cache_size_value,
    kv_cache_precision_value,
)
from tests.functional.constants.generative_ai import GenerativeAIPluginConfig
from tests.functional.constants.ovms import CurrentTarget as ct
from tests.functional.constants.ovms_cohere import OvmsCohereRequestParamsBuilder
from tests.functional.constants.ovms_openai import OvmsOpenAIRequestParamsBuilder
from tests.functional.constants.ovms_messages import OvmsMessages
from tests.functional.fixtures.server import start_ovms
from tests.functional.object_model.inference_helpers import run_llm_inference
from tests.functional.object_model.ovms_params import OvmsParams
from tests.functional.utils.assertions import assert_raises_exception
from tests.functional.utils.context import Context
from tests.functional.utils.hooks import timeout_dict
from tests.functional.utils.inference.serving.cohere import CohereWrapper
from tests.functional.utils.logger import step
from tests.functional.utils.marks import MarkRunType
from tests.functional.object_model.python_custom_nodes.python_custom_nodes import (
    SimpleLLM,
    SimpleFeatureExtraction,
    SimpleRerank,
    SimpleImageGeneration,
    SimpleAsrModel,
    SimpleTtsModel,
)

INITIALIZE_LLM_TIMEOUT = 900


def calculate_generative_test_timeout(lm_time_sec):
    test_timeout_sec = lm_time_sec + INITIALIZE_LLM_TIMEOUT
    return (
        test_timeout_sec
        if test_timeout_sec > timeout_dict[MarkRunType.TEST_MARK_REGRESSION_WEEKLY_SINGLE] or ct.is_gpu_target()
        else timeout_dict[MarkRunType.TEST_MARK_REGRESSION_WEEKLY_SINGLE]
    )


class GenerativeAIUtils:

    @staticmethod
    def prepare_request_params(endpoint, **request_params_kwargs):
        request_params_builder_class = OvmsCohereRequestParamsBuilder if endpoint == CohereWrapper.RERANK \
            else OvmsOpenAIRequestParamsBuilder
        request_params_builder = request_params_builder_class(
            endpoint=endpoint,
            **request_params_kwargs
        )
        request_params = request_params_builder.request_params
        return request_params

    @staticmethod
    def prepare_model(
            model_type,
            kv_cache_size=kv_cache_size_value,
            plugin_config=None,
            max_position_embeddings=None,
            tools_enabled=None,
            apply_gorilla_patch=False,
            enable_tool_guided_generation=False,
            target_device=None,
            resolution=None,
            apply_short_name=False,
            **kwargs
    ):
        if plugin_config is None:
            plugin_config = {GenerativeAIPluginConfig.KV_CACHE_PRECISION: kv_cache_precision_value}

        stream = kwargs.get("stream", False)

        step("Prepare language model instance")
        llm = model_type()
        llm.apply_gorilla_patch = apply_gorilla_patch
        if apply_gorilla_patch:
            llm.name = f"{llm.gorilla_patch_name}-stream" if stream else llm.gorilla_patch_name
        if apply_short_name:
            llm.name = llm.name.split("/")[-1] if "/" in llm.name else llm.name
        if tools_enabled:
            llm.tools_enabled = True

        node_name = "LLMExecutor"
        if hasattr(llm, "is_feature_extraction") and llm.is_feature_extraction:
            node_class = SimpleFeatureExtraction
        elif hasattr(llm, "is_rerank") and llm.is_rerank:
            node_class = SimpleRerank
        elif hasattr(llm, "is_image_generation") and llm.is_image_generation:
            node_class = SimpleImageGeneration
            node_name = "ImageGenExecutor"
        elif hasattr(llm, "is_asr_model") and llm.is_asr_model:
            node_class = SimpleAsrModel
            node_name = "S2tExecutor"
        elif hasattr(llm, "is_tts_model") and llm.is_tts_model:
            node_class = SimpleTtsModel
            node_name = "T2sExecutor"
        else:
            node_class = SimpleLLM

        model = node_class(
            model=llm,
            node_name=node_name,
            models_path=llm.model_path_on_host,
            kv_cache_size=kv_cache_size,
            plugin_config=plugin_config,
            enable_tool_guided_generation=enable_tool_guided_generation,
            target_device=target_device,
            resolution=resolution,
        )
        model.precision = llm.precision
        model.max_position_embeddings = max_position_embeddings
        model.jinja_template = llm.jinja_template
        model.allows_reasoning = llm.allows_reasoning
        model.apply_gorilla_patch = apply_gorilla_patch
        model.gorilla_patch_name = llm.gorilla_patch_name
        model.enable_tool_guided_generation = enable_tool_guided_generation
        model.bfcl_num_threads = llm.bfcl_num_threads
        return model

    @classmethod
    def prepare_resources(
        cls, context: Context, model_type, openai_rest_api_type, endpoint, log_level=logging_level_ovms,
            kv_cache_size=kv_cache_size_value, plugin_config=None, max_position_embeddings=None, env=None,
            allowed_local_media_path=None, allowed_media_domains=None, target_device=None, resolution=None,
            apply_short_name=False, **request_params_kwargs,
    ):
        if plugin_config is None:
            plugin_config = {GenerativeAIPluginConfig.KV_CACHE_PRECISION: kv_cache_precision_value}

        model = cls.prepare_model(
            model_type,
            kv_cache_size,
            plugin_config,
            max_position_embeddings,
            target_device=target_device,
            resolution=resolution,
            apply_short_name=apply_short_name,
        )

        step("Start OVMS")
        result = start_ovms(
            context,
            OvmsParams(models=[model], use_config=True, use_subconfig=model.use_subconfig, log_level=log_level,
                       allowed_local_media_path=allowed_local_media_path, allowed_media_domains=allowed_media_domains),
            timeout=model.model_timeout, environment=env,
        )
        port = result.ovms.get_port(openai_rest_api_type)

        request_params = cls.prepare_request_params(endpoint, **request_params_kwargs)

        return model, result, port, request_params

    @staticmethod
    def unload_model_and_verify(model, result, port, openai_rest_api_type, endpoint, request_parameters,
                                dataset=None, error_type=NotFoundError):
        step("Unload model")
        ovms_log_monitor = result.ovms.create_log(False)
        result.ovms.unload_all_models()
        ovms_log_monitor.models_unloaded([model])
        result.models = []

        step("Verify model is unreachable")
        assert_raises_exception(
            error_type,
            OvmsMessages.MEDIAPIPE_IS_RETIRED,
            run_llm_inference,
            model=model,
            api_type=openai_rest_api_type,
            port=port,
            endpoint=endpoint,
            dataset=dataset,
            request_parameters=request_parameters,
        )
