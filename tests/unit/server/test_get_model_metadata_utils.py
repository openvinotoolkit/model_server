#
# Copyright (c) 2018 Intel Corporation
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

from tensorflow.python.framework import dtypes as dtypes
import numpy as np
import pytest
from ie_serving.server.get_model_metadata_utils import \
    _prepare_signature, prepare_get_metadata_output
from conftest import MockedIOInfo


@pytest.mark.parametrize("layers, tensor_key, np_type", [
    ({'tensor': MockedIOInfo('FP32', (1, 1, 1), 'NCHW'),
      'test_tensor': MockedIOInfo('FP32', (1, 1, 1), 'NCHW')},
     {'new_key': 'tensor', 'client_key': 'test_tensor'}, np.float32),
    ({'tensor': MockedIOInfo('I32', (1, 1, 1), 'NCHW')}, {'new_key': 'tensor'},
     np.int32),
])
def test_prepare_signature(layers, tensor_key, np_type):
    dtype_model = dtypes.as_dtype(np_type)
    output = _prepare_signature(
        layers=layers, model_keys=tensor_key)

    for key, value in tensor_key.items():
        assert key in output
        assert value in output[key].name

        shape = [d.size for d in output[key].tensor_shape.dim]
        assert list(layers[value].shape) == shape

        tensor_dtype = dtypes.as_dtype(output[key].dtype)
        assert dtype_model == tensor_dtype


def test_prepare_get_metadata_output():
    inputs = {'tensor_input': MockedIOInfo('FP32', (1, 1, 1), 'NCHW')}
    outputs = {'tensor_output': MockedIOInfo('FP32', (1, 1, 1), 'NCHW')}
    model_keys = {'inputs': {'name': 'tensor_input'},
                  'outputs': {'output_name': 'tensor_output'}}
    output = prepare_get_metadata_output(
        inputs=inputs, outputs=outputs, model_keys=model_keys)

    assert "tensorflow/serving/predict" == output.method_name
