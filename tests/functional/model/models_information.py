#
# Copyright (c) 2020 Intel Corporation
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


class Resnet:
    name = "resnet"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (1, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (1, 1001)
    rest_request_format = 'column_name'


class ResnetBS4:
    name = "resnet_bs4"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (4, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (4, 1001)
    rest_request_format = 'row_noname'


class ResnetBS8:
    name = "resnet_bs8"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (8, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (8, 1001)
    rest_request_format = 'row_noname'


class ResnetS3:
    name = "resnet_s3"
    dtype = np.float32
    input_name = "map/TensorArrayStack/TensorArrayGatherV3"
    input_shape = (1, 3, 224, 224)
    output_name = "softmax_tensor"
    output_shape = (1, 1001)
    rest_request_format = 'row_name'


class ResnetGS:
    name = "resnet_gs"
    dtype = np.float32
    input_name = "input"
    input_shape = (1, 3, 224, 224)
    output_name = "resnet_v1_50/predictions/Reshape_1"
    output_shape = (1, 1000)
    rest_request_format = 'row_name'
