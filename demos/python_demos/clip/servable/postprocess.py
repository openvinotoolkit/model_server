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
        pass
    
    def execute(self, inputs: list):

        probs = np.frombuffer(inputs[0].data, dtype=inputs[0].datatype)
        input_labels = np.frombuffer(inputs[1].data, dtype=inputs[1].datatype)

        max_prob = probs.argmax(axis=1)
        max_label = input_labels[max_prob]
        return [Tensor("detection", max_label)]

