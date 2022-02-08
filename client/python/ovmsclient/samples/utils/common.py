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
from os import listdir
from os.path import isfile, join


def read_image_paths(images_dir):
    return [join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))]


def load_image(path):
    with open(path, 'rb') as f:
        return f.read()


# get input and output name from model metadata
def get_model_io_names(client, model_name, model_version):
    metadata = client.get_model_metadata(model_name, model_version)
    input_name = next(iter(metadata['inputs']))
    output_name = next(iter(metadata['outputs']))
    return input_name, output_name


# get input shape from model metadata
def get_model_input_shape(client, model_name, model_version):
    metadata = client.get_model_metadata(model_name, model_version)
    inputs = metadata['inputs']
    input_name = next(iter(inputs))
    return inputs[input_name]['shape']
