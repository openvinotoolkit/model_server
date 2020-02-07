#
# Copyright (c) 2018-2020 Intel Corporation
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

from ie_serving.server import predict_utils
import pytest
from tensorflow import make_ndarray
import numpy as np

SERIALIZATION_FUNCTIONS = {
    '_prepare_output_with_make_tensor_proto':
        predict_utils._prepare_output_with_make_tensor_proto,
    '_prepare_output_as_AppendArrayToTensorProto':
        predict_utils._prepare_output_as_AppendArrayToTensorProto
}


@pytest.mark.parametrize("serialization_function",
                         ['_prepare_output_with_make_tensor_proto',
                          '_prepare_output_as_AppendArrayToTensorProto'])
@pytest.mark.parametrize("outputs_names, shapes, types", [
    ({'resnet': 'test'}, [(1, 1)], [np.int32]),
    ({'resnet': 'test'}, [(2, 2)], [np.float32]),
    ({'resnet': 'test'}, [(2, 2, 2)], [np.double]),
    ({'resnet': 'test', 'model': 'tensor'}, [(1, 1), (2, 2)],
     [np.double, np.int32]),
    ({'resnet': 'test', 'model': 'tensor'}, [(3, 4), (5, 6, 7)],
     [np.double, np.int32, np.float32])
])
def test_prepare_output_as_list(serialization_function, outputs_names, shapes,
                                types):
    outputs = {}
    x = 0
    for key, value in outputs_names.items():
        outputs[value] = np.ones(shape=shapes[x], dtype=types[x])
        x += 1

    output = SERIALIZATION_FUNCTIONS[serialization_function](
        inference_output=outputs, model_available_outputs=outputs_names)

    x = 0
    for key, value in outputs_names.items():
        temp_output = make_ndarray(output.outputs[key])
        assert temp_output.shape == shapes[x]
        assert temp_output.dtype == types[x]
        x += 1
