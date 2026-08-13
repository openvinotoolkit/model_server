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

import glob
import json
import argparse
import os
import openvino as ov
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-repo",
    default="OpenVINO/yolox_tiny-fp16-ov",
    help="Hugging Face model repository",
)
args = parser.parse_args()

MODEL_REPO = args.model_repo
MODEL_NAME = MODEL_REPO.split("/")[-1]
# ---------------------------------------------------------
# 1. Download model
# ---------------------------------------------------------

model_dir = snapshot_download(repo_id=MODEL_REPO)

xml_path = glob.glob(os.path.join(model_dir, "*.xml"))[0]
config_path = os.path.join(model_dir, "config.json")

print("Found IR    :", xml_path)
print("Found config:", config_path)


# ---------------------------------------------------------
# 2. Read config.json
# ---------------------------------------------------------

with open(config_path, "r") as f:
    config = json.load(f)

print("\nModel config:")
print("model_name :", config.get("model_name"))
print("model_type :", config.get("model_type"))
print("input_type :", config.get("input_dtype"))
print("mean_values:", config.get("mean_values"))
print("scale_values:", config.get("scale_values"))
print("classes:", config.get("labels"))

# ---------------------------------------------------------
# 3. Prepare classes list
# ---------------------------------------------------------
classes = config.get("labels").split(" ")

# ---------------------------------------------------------
# 4. Parse mean and scale values
# ---------------------------------------------------------
mean_values = [float(x) for x in config["mean_values"].split()]
scale_values = [float(x) for x in config["scale_values"].split()]

print("\nParsed preprocessing:")
print("mean :", mean_values)
print("scale:", scale_values)

# ---------------------------------------------------------
# 5. Load OpenVINO model
# ---------------------------------------------------------
core = ov.Core()
model = core.read_model(xml_path)
# ---------------------------------------------------------
# 6. Configure preprocessing
# ---------------------------------------------------------
ppp = ov.preprocess.PrePostProcessor(model)
inp = ppp.input(0)

# Input coming from user/image:
# f32 NHWC
inp.tensor().set_element_type(ov.Type.f32).set_layout(ov.Layout("NHWC"))

# Model expects:
# float32 NCHW
inp.model().set_layout(ov.Layout("NCHW"))

# Preprocessing:

inp.preprocess().convert_element_type(ov.Type.f32).convert_layout(
    ov.Layout("NCHW")
).scale(255.0).mean(mean_values).scale(scale_values)
# ---------------------------------------------------------
# 7. Build and save
# ---------------------------------------------------------

model = ppp.build()

output_path = f"{MODEL_NAME}/1/{MODEL_NAME}.xml"

ov.save_model(model, output_path)

print("\nSaved:", os.path.abspath(output_path))

with open("coco_80cl.txt", "w") as f:
    n = len(classes)
    for i, c in enumerate(classes):
        f.write(c + ("\n" if i < n - 1 else ""))

print("Downloaded successfully")
