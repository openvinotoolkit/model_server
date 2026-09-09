#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
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

import openvino_genai as genai
pipe = genai.LLMPipeline('ov_model', 'CPU')
print(pipe.generate('Hello', max_new_tokens=1000, ignore_eos=True, do_sample=False))

pipe = genai.VLMPipeline('vlm_ov_model', 'CPU')
print(pipe.generate('Hello', max_new_tokens=1000, ignore_eos=True, do_sample=False))
