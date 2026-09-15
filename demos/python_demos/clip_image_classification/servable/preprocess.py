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
from tritonclient.utils import deserialize_bytes_tensor

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        model_id = "openai/clip-vit-base-patch16"
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def execute(self, inputs: list):
        image_raw = deserialize_bytes_tensor(bytes(inputs[0]))[0]
        labels_raw = deserialize_bytes_tensor(bytes(inputs[1]))[0]
        image = Image.open(BytesIO(image_raw))
        input_labels = labels_raw.decode()
        input_labels_split = [label.strip() for label in input_labels.split(",") if label.strip()]
        if not input_labels_split:
            input_labels_split = ["item"]
        text_descriptions = [f"This is a photo of a {label}" for label in input_labels_split]

        # Split processing avoids combined CLIPProcessor batching edge-cases in runtime packaging.
        text_inputs = self.processor.tokenizer(
            text_descriptions,
            return_tensors="np",
            padding=True,
            truncation=True,
        )
        image_inputs = self.processor.image_processor(images=image.convert("RGB"), return_tensors="np")

        # Explicit INT64 is required because on some platforms NumPy int64 can infer as "l" -> INT32 in pyovms.
        input_ids = np.array(text_inputs["input_ids"], dtype=np.int64)
        attention_mask = np.array(text_inputs["attention_mask"], dtype=np.int64)
        pixel_values = np.array(image_inputs["pixel_values"], dtype=np.float32)

        input_ids_py = Tensor("input_ids_py", input_ids, datatype="INT64")
        attention_mask_py = Tensor("attention_mask_py", attention_mask, datatype="INT64")
        pixel_values_py = Tensor("pixel_values_py", pixel_values, datatype="FP32")

        return [input_ids_py, attention_mask_py, pixel_values_py]

