#*****************************************************************************
# Copyright 2023 Intel Corporation
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
#*****************************************************************************

import tritonclient.grpc as grpcclient
from tritonclient.utils import serialize_byte_tensor
import numpy as np

def serialize_prompts(prompts):
    infer_input = grpcclient.InferInput("pre_prompt", [len(prompts)], "BYTES")
    infer_input._raw_content = serialize_byte_tensor(
        np.array(prompts, dtype=np.object_)).item()
    return infer_input
