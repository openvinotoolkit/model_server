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

from pyovms import Tensor
from transformers import CLIPProcessor, CLIPModel
from urllib.request import urlretrieve
from pathlib import Path
from PIL import Image
import numpy as np
import os
import openvino as ov
from scipy.special import softmax

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        model_id = "openai/clip-vit-base-patch16"
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def execute(self, inputs: list):
        input_url = bytes(inputs[0]).decode()
        input_name = input_url.split("/")[-1]
        sample_path = Path(os.path.join("data", input_name))
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            input_url,
            sample_path,
        )
        image = Image.open(sample_path)

        input_labels = np.frombuffer(inputs[1].data, dtype=inputs[1].datatype)
        self.text_descriptions = [f"This is a photo of a {label}" for label in input_labels]
        model_inputs = self.processor(text=self.text_descriptions, images=[image], return_tensors="pt", padding=True)

        input_ids_py = Tensor("input_ids_py", model_inputs["input_ids"].numpy().astype(np.int64))
        attention_mask_py = Tensor("attention_mask_py", model_inputs["attention_mask"].numpy().astype(np.int64))
        pixel_values_py = Tensor("pixel_values_py", model_inputs["pixel_values"].numpy())
        print("input_ids_py " + input_ids_py.datatype)
        print("attention_mask_py " + attention_mask_py.datatype)
        print("pixel_values_py " + pixel_values_py.datatype)
        return [input_ids_py, attention_mask_py, pixel_values_py]

