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
        model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        # create OpenVINO core object instance
        core = ov.Core()
        device = "CPU"
        model.config.torchscript = True
        input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
        text_descriptions = [f"This is a photo of a {label}" for label in input_labels]
        image =  Image.new('RGB', (800, 600))
        model_inputs = self.processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)

        ov_model = ov.convert_model(model, example_input=dict(model_inputs))
        
        # compile model for loading on device
        self.compiled_model = core.compile_model(ov_model, device)
        # obtain output tensor for getting predictions
        self.logits_per_image_out = self.compiled_model.output(0)

    def execute(self, inputs: list):
        model_inputs = np.frombuffer(inputs[0].data, dtype=inputs[0].datatype)
        logits_per_image = self.compiled_model(dict(model_inputs))[self.logits_per_image_out]

        probs = softmax(logits_per_image, axis=1)
        return [Tensor("logits_per_image", probs)]

