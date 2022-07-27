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

import sys
sys.path.append("../../../../demos/common/python")

import numpy as np
from classes import imagenet_classes

# get input and output name from model metadata
def get_model_io_names(client, model_name, model_version):
    metadata = client.get_model_metadata(model_name, model_version)
    if len(metadata['inputs']) > 1:
        raise ValueError("Unexpected multiple model inputs")
    input_name = next(iter(metadata['inputs']))  # by default resnet has only one input and one output in shape (1,1000) or (1,1001)
    output_name = None
    for name, meta in metadata['outputs'].items():
        if meta["shape"] in [[1,1001], [1, 1000], [-1,1000], [-1,1001]]:
            if output_name is not None:
                raise ValueError("Unexpected multiple models outputs with shapes in [(1,1001), (1, 1000), (-1,1001), (-1, 1000)]")
            output_name = name
    if output_name is None:
        raise ValueError("Could not find output with shape (1,1001), (1, 1000), (-1,1001) or (-1, 1000) among model outputs")
    return input_name, output_name

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
