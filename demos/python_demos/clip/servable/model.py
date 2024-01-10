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

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        model_id = "openai/clip-vit-base-patch16"
        self.model = CLIPModel.from_pretrained(model_id)
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
        results = self.model(**model_inputs)
        logits_per_image = results['logits_per_image']
        probs = logits_per_image.softmax(dim=1).detach().numpy()
        return [Tensor("logits_per_image", probs)]

