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

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        model_id = "openai/clip-vit-base-patch16"
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        sample_path = Path("data/coco.jpg")
        if not sample_path.exists():
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(
                "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
                sample_path,
            )
        self.image = Image.open(sample_path)

        input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
        self.text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

    def execute(self, inputs: list):
        inputs = self.processor(text=self.text_descriptions, images=[self.image], return_tensors="pt", padding=True)
        results = self.model(**inputs)
        logits_per_image = results['logits_per_image']
        return [Tensor("logits_per_image", logits_per_image.encode())]

