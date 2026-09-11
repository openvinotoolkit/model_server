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

from tests.functional.models.models_library import ModelsLib
from tests.functional.constants.components import OvmsComponents
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
from tests.functional.utils.inference.serving.openai import OpenAIWrapper
from tests.functional.utils.logger import get_logger, step
from tests.functional.utils.test_framework import (
    skip_if_language_models_not_enabled,
    skip_if_mediapipe_disabled,
)

logger = get_logger(__name__)


@pytest.mark.priority_high
@pytest.mark.components(OvmsComponents.OVMS)
@pytest.mark.reqids(Requirements.rerank_endpoint, Requirements.openai_api)
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
    @pytest.mark.timeout(calculate_generative_test_timeout(480))
    def test_on_commit_llm_text_generation_endpoints(
            self, context: Context, model_type, openai_rest_api_type, endpoint, stream, max_tokens, temperature
    ):
        """
        <b>Description:</b>
        Execute single inference with LLM/VLM type model.

        <b>Input data:</b>
        - Language model type/ Vision Language model type

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
            model_type,
            openai_rest_api_type,
            endpoint,
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
