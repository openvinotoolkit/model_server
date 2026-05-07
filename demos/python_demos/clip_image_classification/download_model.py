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

from optimum.intel import OVModelForZeroShotImageClassification
import os

model_id = "openai/clip-vit-base-patch16"
model_path = "model/1"

print(f"Exporting {model_id} to OpenVINO IR via optimum-intel ...")
ov_model = OVModelForZeroShotImageClassification.from_pretrained(model_id, export=True)

os.makedirs(model_path, exist_ok=True)
ov_model.save_pretrained(model_path)

# OVMS expects flat <name>.xml/.bin in the version directory.
# optimum-intel saves them as openvino_model.xml/.bin; rename to keep the existing graph.pbtxt working.
for src, dst in (
    ("openvino_model.xml", "clip-vit-base-patch16.xml"),
    ("openvino_model.bin", "clip-vit-base-patch16.bin"),
):
    src_path = os.path.join(model_path, src)
    if os.path.exists(src_path):
        os.replace(src_path, os.path.join(model_path, dst))

print("Model ready")
