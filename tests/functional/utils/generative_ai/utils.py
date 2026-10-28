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

import json
import os
import sys
import numpy as np

from openai import NotFoundError

from tests.functional.config import (
    logging_level_ovms,
    kv_cache_size_value,
    kv_cache_precision_value,
    ovms_c_repo_path,
)
from tests.functional.constants.generative_ai import GenerativeAIPluginConfig
from tests.functional.constants.os_type import OsType
from tests.functional.constants.ovms import CurrentTarget as ct
from tests.functional.constants.ovms_cohere import OvmsCohereRequestParamsBuilder
from tests.functional.constants.ovms_openai import OvmsOpenAIRequestParamsBuilder
from tests.functional.constants.ovms_messages import OvmsMessages
from tests.functional.constants.paths import Paths
from tests.functional.fixtures.server import start_ovms
from tests.functional.models.models_datasets import FeatureExtractionModelDataset
from tests.functional.models.models_generative import Qwen3Embedding06BFp16OvHf
from tests.functional.object_model.inference_helpers import run_llm_inference
from tests.functional.object_model.mediapipe_calculators import MediaPipeCalculator
from tests.functional.object_model.ovms_config import OvmsConfig
from tests.functional.object_model.ovms_params import OvmsParams
from tests.functional.object_model.python_custom_nodes.python_custom_nodes import (
    SimpleLLM,
    SimpleFeatureExtraction,
    SimpleRerank,
    SimpleImageGeneration,
    SimpleAsrModel,
    SimpleTtsModel,
)
from tests.functional.object_model.test_environment import TestEnvironment
from tests.functional.object_model.test_helpers import send_reload_request
from tests.functional.utils.assertions import assert_raises_exception
from tests.functional.utils.context import Context
from tests.functional.utils.hooks import timeout_dict
from tests.functional.utils.inference.serving.cohere import CohereWrapper
from tests.functional.utils.inference.serving.openai import OpenAIWrapper
from tests.functional.utils.logger import get_logger, step
from tests.functional.utils.marks import MarkRunType

logger = get_logger(__name__)

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
            enable_word_timestamps=False,
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
            enable_word_timestamps=enable_word_timestamps,
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
            apply_short_name=False, enable_word_timestamps=False, **request_params_kwargs,
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
            enable_word_timestamps=enable_word_timestamps,
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

    @staticmethod
    def prepare_python_script_command(
            context, tmp_path, port, model, endpoint, input_data, request_parameters, image_path=None, timeout=900
    ):
        input_data_json_path = os.path.join(tmp_path, "input_data.json")
        with open(input_data_json_path, "w", encoding='utf-8') as fo:
            json.dump(input_data, fo)
        # sys.prefix follows whichever repo the tests were started from
        venv_cmd = os.path.join(sys.prefix, "Scripts", "activate.bat") if context.base_os == OsType.Windows else \
            f"source {os.path.join(sys.prefix, 'bin', 'activate')}"
        script_path = os.path.join(
            ovms_c_repo_path, "tests", "functional", "utils", "generative_ai", "openai_client.py")
        request_params_dict = request_parameters.prepare_dict(use_extra_body=False)
        request_params_str = " ".join(f"--{key} {value}" for key, value in request_params_dict.items())
        cmd = (f"{venv_cmd} && python3 {script_path} --port {port} --model {model.name} --endpoint {endpoint} "
               f"--input_data_file {input_data_json_path} {request_params_str} --timeout {timeout}")
        if image_path is not None:
            cmd += f" --image_path {image_path}"
        return cmd

    @staticmethod
    def get_llm_outputs_from_curl_stdout(stdout):
        braces_left_str = "[\""
        braces_right_str = "\"]"
        stdout_list = stdout.replace("['", braces_left_str).replace("']", braces_right_str).split(braces_right_str)
        outputs_multiple_requests = [
            [(outputs_str + braces_right_str).strip().strip(braces_left_str).strip(braces_right_str)]
            for outputs_str in stdout_list[:-1]
        ]
        return outputs_multiple_requests

    @staticmethod
    def load_llm_model_and_verify(model, result, context, port, reload_retry=1):
        result.models.append(model)
        result.ovms.prepare_resources([model])
        result.ovms.update_model_list_and_config(result.ovms.name, result.models, context=context)

        params = OvmsParams(models=[model], use_config=True, use_subconfig=model.use_subconfig)
        config_dir_path_on_host = os.path.join(
            TestEnvironment.current.base_dir,
            result.ovms.name,
            Paths.MODELS_PATH_NAME,
        )
        if model.use_subconfig:
            OvmsConfig.create_subconfig(result.ovms.name, params, config_dir_path_on_host)
        MediaPipeCalculator.prepare_proto_calculator(
            params,
            config_dir_path_on_host,
            os.path.join(config_dir_path_on_host, Paths.CONFIG_FILE_NAME)
        )

        ovms_log = result.ovms.create_log(False)
        send_reload_request(port, timeout=300, retry=reload_retry, retry_delay=2, retry_backoff=2, retry_max_delay=15)
        break_msg_list = result.ovms.get_break_msg_list([model])
        ovms_log.models_loaded([model], break_msg_list=break_msg_list, ovms_instance=result.ovms)

    @classmethod
    def check_llm_outputs_similarity(
            cls,
            context,
            outputs_list,
            reference_outputs,
            result,
            openai_rest_api_type,
            port,
            cos_sim_reference=0.7,
            embeddings_model_type=Qwen3Embedding06BFp16OvHf,
    ):
        step("Check Similarity")
        embeddings_model = GenerativeAIUtils.prepare_model(embeddings_model_type)
        cls.load_llm_model_and_verify(embeddings_model, result, context, port)

        embeddings_request_params = GenerativeAIUtils.prepare_request_params(OpenAIWrapper.EMBEDDINGS)
        embedding_from_string_reference = None
        cos_sim_errors = []
        for index, outputs in enumerate([reference_outputs] + outputs_list):

            class TestDataset(FeatureExtractionModelDataset):
                input_data = [outputs[0]]

            _, embeddings_outputs_raw, _, _ = run_llm_inference(
                embeddings_model,
                openai_rest_api_type,
                port,
                OpenAIWrapper.EMBEDDINGS,
                dataset=TestDataset,
                input_data_type="list",
                request_parameters=embeddings_request_params,
            )

            if index == 0:
                embedding_from_string_reference = np.array(embeddings_outputs_raw.data[0].embedding)
            else:
                embedding_from_string = np.array(embeddings_outputs_raw.data[0].embedding)
                cos_sim = np.dot(embedding_from_string_reference, embedding_from_string) / \
                          (np.linalg.norm(embedding_from_string_reference) * np.linalg.norm(embedding_from_string))
                logger.debug(f"Similarity score as cos_sim: {cos_sim}")
                if cos_sim < cos_sim_reference:
                    cos_sim_errors.append({"cos_sim": cos_sim, "outputs": outputs[0]})

        assert not cos_sim_errors, \
            f"Similarity score is too low. Reference outputs: {reference_outputs[0]}. Results: {cos_sim_errors}"
