#
# Copyright (c) 2021 Intel Corporation
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

import numpy as np
from imagenet_classes import imagenet_classes


def resnet_postprocess(response, output_name):
    if isinstance(response, dict):
        output = response[output_name]
    else:
        output = response
    
    predicted_class = np.argmax(output[0])
    offset = 0
    if output.shape[1] == 1001:
        offset = 1
    confidence_score = output[0][predicted_class]
    label = imagenet_classes[predicted_class-offset]
    return label, confidence_score
