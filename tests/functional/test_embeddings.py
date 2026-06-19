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

import pytest

from tests.functional.models.models_library import ModelsLib
from tests.functional.constants.components import OvmsComponents
from tests.functional.constants.ovms_openai import EncodingFormatValues
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
@pytest.mark.reqids(Requirements.embeddings_endpoint, Requirements.openai_api)
@pytest.mark.ovms_types_supported_for_test(
    OvmsType.DOCKER,
    OvmsType.DOCKER_CMD_LINE,
    OvmsType.BINARY,
    OvmsType.BINARY_DOCKER,
)
@skip_if_language_models_not_enabled()
@nginx_mtls_not_supported_for_test()
@skip_if_mediapipe_disabled()
class TestEmbeddings:

    @staticmethod
    def run_embeddings_endpoints_test(
            context: Context, model_type, openai_rest_api_type, endpoint, encoding_format, input_data_type
    ):
        model, result, port, request_params = GenerativeAIUtils.prepare_llm_resources(
            context,
            model_type,
            openai_rest_api_type,
            endpoint,
            encoding_format=encoding_format,
        )

        step("Run simple inference")
        run_llm_inference(
            model,
            openai_rest_api_type,
            port,
            endpoint,
            input_data_type=input_data_type,
            request_parameters=request_params,
        )

        GenerativeAIUtils.unload_llm_model_and_verify(
            model,
            result,
            port,
            openai_rest_api_type,
            endpoint,
            request_params
        )

    @pytest.mark.api_on_commit
    @pytest.mark.devices_supported_for_test(TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU)
    @pytest.mark.model_type(ModelsLib.various_feature_extraction_models)
    @pytest.mark.parametrize("endpoint", [OpenAIWrapper.EMBEDDINGS])
    @pytest.mark.parametrize("encoding_format", EncodingFormatValues.values(), ids=lambda x: f"encoding_format={x}")
    @pytest.mark.parametrize("input_data_type", ["list", "string"], ids=lambda x: f"input_data_type={x}")
    @pytest.mark.timeout(calculate_generative_test_timeout(480))
    def test_on_commit_embeddings_endpoints(
            self, context: Context, model_type, openai_rest_api_type, endpoint, encoding_format, input_data_type
    ):
        """
        <b>Description:</b>
        Execute single inference with LLM type model using embeddings endpoint.

        <b>Input data:</b>
        - Language model (feature extraction) type

        <b>Expected results:</b>
        OVMS will properly load language model and execute inference.

        <b>Steps:</b>
        1. Prepare language model instance
        2. Start OVMS
        3. Run simple inference
        4. Unload model
        5. Verify model is unreachable
        """
        self.run_embeddings_endpoints_test(
            context, model_type, openai_rest_api_type, endpoint, encoding_format, input_data_type
        )
