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
from transformers import CLIPProcessor
from PIL import Image
import numpy as np
from io import BytesIO

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        model_id = "openai/clip-vit-base-patch16"
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def execute(self, inputs: list):
        print(inputs[0].datatype)
        print(inputs[1].datatype)
        #image = Image.open(BytesIO(inputs[0].data))
        image_buffer = np.array(inputs[0].data, dtype=np.object_)
        #image_buffer = image_buffer
        print(np.uint8(image_buffer).shape)
        image = Image.fromarray(np.uint8(image_buffer)[4:]).convert('RGB')
        #print(list(image.getdata()))
        
        input_labels = np.array(inputs[1].data, dtype=np.uint8).tobytes().decode("utf-8")[4:]
        #print(input_labels)

        input_labels_split = input_labels.split(",")
        print(input_labels_split)

        text_descriptions = [f"This is a photo of a {label}" for label in input_labels_split]
        
        #error1 = int(1).split("o")
        print("processor")
        model_inputs = self.processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)

        # dtype=np.dtype("q") must be used for correct mapping of struct format character to datatype
        # using np.int64 sets format character to "l" which translates to INT32 in Tensor.datatype
        input_ids = np.array(model_inputs["input_ids"], dtype=np.dtype("q"))
        attention_mask = np.array(model_inputs["attention_mask"], dtype=np.dtype("q"))

        input_ids_py = Tensor("input_ids_py", input_ids)
        attention_mask_py = Tensor("attention_mask_py", attention_mask)
        pixel_values_py = Tensor("pixel_values_py", model_inputs["pixel_values"].numpy())
        print("DONE")
        return [input_ids_py, attention_mask_py, pixel_values_py]

