#!/bin/bash
#
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
#
python3 -m transformers.onnx \
    --feature causal-lm \
    --atol 1e-04 \
    --preprocessor tokenizer \
    --model=local-pt-checkpoint/ \
    --framework pt \
    onnx/1

# Desired output:
# Validating ONNX model...
#         -[✓] ONNX model output names match reference model ({'logits'})
#         - Validating ONNX Model output "logits":
#                 -[✓] (3, 9, 50400) matches (3, 9, 50400)
#                 -[✓] all values close (atol: 0.0001)
# All good, model saved at: onnx/1/model.onnx
