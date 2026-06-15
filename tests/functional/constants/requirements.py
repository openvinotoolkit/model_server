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

class Requirements:

    # model types
    onnx = "CVS-29667 ONNX"

    # model operations
    status = "CVS-35266_CVS-29674_CVS-30289 get model status"
    metadata = "CVS-35266_CVS-29674_CVS-30289 get model metadata"

    # features
    parity = "CVS-35266 parity"
    model_control_api = "CVS-40334 model control API"
    online_modification = "CVS-33847 online model modification"
    custom_loader = "CVS-40416 custom loader"
    nginx = "CVS-40416 nginx"
    dag = "CVS-31796_CVS-36434_CVS-41115 DAG"
    model_cache = "CVS-62829 model_cache"
    cloud = "CVS-31243 cloud"
    stateful = "CVS-33882 stateful"
    reshape = "CVS-35266 reshape"
    dynamic_shape = "CVS-56655 dynamic shapes"
    auto_plugin = "CVS-73689 Auto plugin support"
    kfservin_api = "CVS-81053 KFServing api"
    metrics = "CVS-43549 metrics"
    custom_nodes = "CVS-44359 custom nodes"
    audio_endpoint = "CVS-174282 audio endpoint"

    # test types
    sdl = "CVS-59335 SDL"
    benchmarks = "CVS-35094 benchmarks"
    example_client = "CVS-35266 example client apps"
    documentation = "CVS-35266 documentation"
    binary_input = "CVS-30320 binary input format"
    cpu_extension = "CVS-68750 cpu extension"

    operator = "CVS-56873 operator"

    models_enabling = "CVS-105320 models enabling"
    triton_async = "CVS-114801 triton async"

    scalar_inputs = "CVS-118200 scalar inputs"

    valgrind = "CVS-125781 valgrind"
    cliloader = "CVS-130322 opencl traces"

    streaming_api = "CVS-118064 streaming API extension"
    mediapipe = "CVS-103194 mediapipe"
    python_custom_node = "CVS-117210 python support"
    llm = "CVS-129298 LLM execution in ovms based on c++ code only"
    openai_api = "CVS-138033 OpenAI API in OVMS"
    hf_imports = "CVS-162541 Direct models import from HF Hub in OVMS"
    tools = "CVS-166514 Structured response with tools support in chat/completions"
