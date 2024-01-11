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
import numpy as np
from scipy.special import softmax

class OvmsPythonModel:

    def initialize(self, kwargs: dict):
        pass
    
    def execute(self, inputs: list):
        ov_logits_per_image = np.asarray(inputs[0].data)
        print("input shape " + str(inputs[0].shape) )
        print("ov_logits_per_image shape" + str(ov_logits_per_image.shape))
        print("ov_logits_per_image " + str(ov_logits_per_image))
        probs = softmax(ov_logits_per_image, axis=1)[0]
        print("probs " + str(probs))
        input_labels = np.frombuffer(inputs[1].data, inputs[1].datatype)

        max_prob = probs.argmax(axis=0)
        max_label = input_labels[max_prob]
        print("max label " + str(max_label))
        print("max label type " + str(type(max_label)))
        print(max_label.data.format)
        max_label = str(max_label)

        return [Tensor("detection", max_label.encode())]

