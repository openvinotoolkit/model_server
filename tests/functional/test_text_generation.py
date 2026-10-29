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

# pylint: disable=too-many-positional-arguments

import pytest

from tests.functional.models.models import PipelineType, model_with_pipeline_type
from tests.functional.models.models_datasets import (
    LongPromptLanguageModelDataset,
    LongPromptSingleMessageLanguageModelDataset,
)
from tests.functional.models.models_library import ModelsLib
from tests.functional.constants.components import OvmsComponents
from tests.functional.constants.ovms import Ovms
from tests.functional.constants.ovms_openai import (
    MaxTokensValues,
    TemperatureValues,
)
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.requirements import Requirements
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.target_device_configuration import nginx_mtls_not_supported_for_test
from tests.functional.object_model.inference_helpers import run_llm_inference
from tests.functional.utils.context import Context
from tests.functional.utils.generative_ai.utils import calculate_generative_test_timeout, GenerativeAIUtils
from tests.functional.utils.inference.serving.openai import OpenAIWrapper, ChatCompletionsApi, CompletionsApi
from tests.functional.utils.logger import get_logger, step
from tests.functional.utils.process_multi import run_command_in_parallel
from tests.functional.utils.test_framework import (
    skip_if_language_models_not_enabled,
    skip_if_mediapipe_disabled,
)

logger = get_logger(__name__)


@pytest.mark.priority_high
@pytest.mark.components(OvmsComponents.OVMS)
@pytest.mark.reqids(Requirements.llm, Requirements.openai_api)
@pytest.mark.ovms_types_supported_for_test(
    OvmsType.DOCKER,
    OvmsType.DOCKER_CMD_LINE,
    OvmsType.BINARY,
    OvmsType.BINARY_DOCKER,
)
@skip_if_language_models_not_enabled()
@nginx_mtls_not_supported_for_test()
@skip_if_mediapipe_disabled()
class TestTextGeneration:

    @pytest.mark.api_on_commit
    @pytest.mark.devices_supported_for_test(TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU)
    @pytest.mark.model_type(ModelsLib.various_large_and_vision_language_models_on_commit)
    @pytest.mark.parametrize("endpoint", OpenAIWrapper.AVAILABLE_TEXT_GENERATION_ENDPOINTS)
    @pytest.mark.parametrize("stream", [True, False], ids=lambda x: f"stream={x}")
    @pytest.mark.parametrize("max_tokens", [MaxTokensValues.DEFAULT], ids=lambda x: f"max_tokens={x}")
    @pytest.mark.parametrize("temperature", [TemperatureValues.TEST_DEFAULT], ids=lambda x: f"temperature={x}")
    @pytest.mark.parametrize("ovms_log_level", [Ovms.LOG_LEVEL_INFO, Ovms.LOG_LEVEL_DEBUG, Ovms.LOG_LEVEL_TRACE])
    @pytest.mark.parametrize("pipeline_type", PipelineType.TEXT_GENERATION)
    @pytest.mark.timeout(calculate_generative_test_timeout(480))
    def test_on_commit_llm_text_generation_endpoints(
            self, context: Context, model_type, openai_rest_api_type, endpoint, stream, max_tokens, temperature,
            pipeline_type, ovms_log_level,
    ):
        """
        <b>Description:</b>
        Execute single inference with LLM/VLM type model.

        <b>Input data:</b>
        - Language model type/ Vision Language model type
        - Pipeline type

        <b>Expected results:</b>
        OVMS will properly load language model and execute inference

        <b>Steps:</b>
        1. Prepare language model instance
        2. Start OVMS
        3. Run simple inference
        4. Unload model
        5. Verify model is unreachable
        """
        model, result, port, request_params = GenerativeAIUtils.prepare_resources(
            context,
            model_with_pipeline_type(model_type, pipeline_type),
            openai_rest_api_type,
            endpoint,
            log_level=ovms_log_level,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        step("Run simple inference")
        run_llm_inference(
            model,
            openai_rest_api_type,
            port,
            endpoint,
            validate_outputs_ttr=False,
            request_parameters=request_params,
        )

        GenerativeAIUtils.unload_model_and_verify(model, result, port, openai_rest_api_type, endpoint, request_params)

    @pytest.mark.api_on_commit
    @pytest.mark.devices_supported_for_test(TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU)
    @pytest.mark.reqids(Requirements.llm, Requirements.openai_api, Requirements.sdl)
    @pytest.mark.model_type(ModelsLib.various_large_and_vision_language_models_on_commit)
    @pytest.mark.parametrize("endpoint", [OpenAIWrapper.CHAT_COMPLETIONS])
    @pytest.mark.parametrize("stream", [True, False], ids=lambda x: f"stream={x}")
    @pytest.mark.parametrize("max_tokens", [MaxTokensValues.COMPARE], ids=lambda x: f"max_tokens={x}")
    @pytest.mark.parametrize("temperature", [TemperatureValues.TEST_DEFAULT], ids=lambda x: f"temperature={x}")
    @pytest.mark.parametrize("sequential_batches", [1], ids=lambda x: f"sequential_batches={x}")
    @pytest.mark.parametrize("parallel_requests", [1000], ids=lambda x: f"parallel_requests={x}")
    @pytest.mark.parametrize("ovms_log_level", [Ovms.LOG_LEVEL_INFO, Ovms.LOG_LEVEL_DEBUG, Ovms.LOG_LEVEL_TRACE])
    @pytest.mark.parametrize("pipeline_type", PipelineType.TEXT_GENERATION)
    # @pytest.mark.timeout(calculate_generative_test_timeout(2200))
    @pytest.mark.timeout(calculate_generative_test_timeout(22000))
    def test_text_generation_multiple_requests(
            self,
            context: Context,
            tmp_path,
            model_type,
            openai_rest_api_type,
            stream,
            endpoint,
            max_tokens,
            temperature,
            sequential_batches,
            parallel_requests,
            pipeline_type,
            ovms_log_level,
    ):
        """
        <b>Description:</b>
        Execute multiple inferences with LLM/VLM type model.

        <b>Input data:</b>
        - Language model type/ Vision Language model type

        <b>Expected results:</b>
        OVMS will properly load language model and execute multiple inferences

        <b>Steps:</b>
        1. Prepare language model instance
        2. Start OVMS
        3. Run simple inference for reference (non-dummy models)
        4. Trigger large number of requests
        5. Check similarity (non-dummy models)
        """
        model, result, port, request_params = GenerativeAIUtils.prepare_resources(
            context,
            model_with_pipeline_type(model_type, pipeline_type),
            openai_rest_api_type,
            endpoint,
            log_level=ovms_log_level,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if model_type.is_vision_language:
            dataset = model_type().get_default_dataset()
        elif model_type.single_message_dataset:
            dataset = LongPromptSingleMessageLanguageModelDataset
        else:
            dataset = LongPromptLanguageModelDataset

        reference_outputs = None
        if not model_type.is_dummy:
            step("Run simple inference for reference (non-dummy models)")
            reference_outputs, _, _, _ = run_llm_inference(
                model,
                openai_rest_api_type,
                port,
                endpoint,
                dataset=dataset,
                request_parameters=request_params,
            )

        step("Trigger large number of requests")
        input_objects = model.prepare_input_data(dataset=dataset)
        input_data = CompletionsApi.prepare_completions_input_content(input_objects) \
            if endpoint == OpenAIWrapper.COMPLETIONS else \
            ChatCompletionsApi.prepare_chat_completions_input_content(input_objects)
        client_timeout = 1800
        command = GenerativeAIUtils.prepare_python_script_command(
            context, tmp_path, port, model, endpoint, input_data, request_params, timeout=client_timeout
        )
        outputs_multiple_requests = None
        for _ in range(sequential_batches):
            stdout = run_command_in_parallel(command, runs=parallel_requests, timeout=client_timeout)
            assert stdout, f"Empty stdout for {parallel_requests} commands: '{command}'"
            outputs_multiple_requests = GenerativeAIUtils.get_llm_outputs_from_curl_stdout(stdout)
            len_stdout = len(outputs_multiple_requests)
            assert len_stdout == parallel_requests, \
                f"Not all outputs available. Expected: {parallel_requests}, Actual: {len_stdout}"

        GenerativeAIUtils.unload_model_and_verify(model, result, port, openai_rest_api_type, endpoint, request_params)
        if not model_type.is_dummy:
            step("Check similarity (non-dummy models)")
            GenerativeAIUtils.check_llm_outputs_similarity(
                context,
                outputs_multiple_requests,
                reference_outputs,
                result,
                openai_rest_api_type,
                port,
            )
