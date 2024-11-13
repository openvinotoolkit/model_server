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

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import openvino as ov
import os

model_id = "openai/clip-vit-base-patch16"
print(f'Downloading pretrained model {model_id} ...')

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# create OpenVINO core object instance
core = ov.Core()
model.config.torchscript = True
input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
text_descriptions = [f"This is a photo of a {label}" for label in input_labels]
image =  Image.new('RGB', (800, 600))
model_inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)

print(f'Converting pretrained model {model_id} ...')
ov_model = ov.convert_model(model, example_input=dict(model_inputs))
for idx, out in enumerate(ov_model.outputs):
    out.get_tensor().set_names({f"out_{idx}"})
print(f'Saving converted model {model_id} ...')
ov.save_model(ov_model, 'clip-vit-base-patch16.xml')

print(f'Creating OpenVINO Model Server model directories ...')
model_path = "model/1"
if not os.path.exists(model_path):
    os.mkdir("model")
    os.mkdir(model_path)

os.replace("clip-vit-base-patch16.bin", os.path.join(model_path,"clip-vit-base-patch16.bin"))
os.replace("clip-vit-base-patch16.xml", os.path.join(model_path,"clip-vit-base-patch16.xml"))

print(f'Model ready')
