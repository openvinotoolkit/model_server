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

from collections import defaultdict

from tests.functional.constants.target_device import TargetDevice
from tests.functional.models.models_generative import (
    BgeRerankerBaseFp16OvHf,
    Gemma34bItInt4OvHf,
    Gemma34bItInt4CwOvHf,
    LFM25350MInt8OvHf,
    Phi35MiniInstructInt4CwOvHf,
    Qwen3Embedding06BFp16OvHf,
    Qwen3Reranker06BFp16OvHf,
    Qwen3Reranker06BSeqClsFp16OvHf,
)


class ModelsLibrary:

    @property
    def various_mini_large_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [LFM25350MInt8OvHf],
                TargetDevice.GPU: [LFM25350MInt8OvHf],
                TargetDevice.NPU: [Phi35MiniInstructInt4CwOvHf],
            },
        )

    @property
    def various_mini_vision_language_models(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU: [Gemma34bItInt4OvHf],
                TargetDevice.GPU: [Gemma34bItInt4OvHf],
                TargetDevice.NPU: [Gemma34bItInt4CwOvHf],
            },
        )

    @property
    def various_large_and_vision_language_models_on_commit(self):
        return defaultdict(
            list,
            {
                TargetDevice.CPU:
                    self.various_mini_large_language_models[TargetDevice.CPU] +
                    self.various_mini_vision_language_models[TargetDevice.CPU],
                TargetDevice.GPU:
                    self.various_mini_large_language_models[TargetDevice.GPU] +
                    self.various_mini_vision_language_models[TargetDevice.GPU],
                TargetDevice.NPU:
                    self.various_mini_large_language_models[TargetDevice.NPU] +
                    self.various_mini_vision_language_models[TargetDevice.NPU],
            },
        )

    @property
    def various_feature_extraction_models_on_commit(self):
        return [Qwen3Embedding06BFp16OvHf]

    @property
    def various_rerank_models_on_commit(self):
        return [BgeRerankerBaseFp16OvHf]

    @property
    def various_rerank_models(self):
        return [
            Qwen3Reranker06BFp16OvHf,
            Qwen3Reranker06BSeqClsFp16OvHf,
        ]


ModelsLib = ModelsLibrary()     # pylint: disable=invalid-name
