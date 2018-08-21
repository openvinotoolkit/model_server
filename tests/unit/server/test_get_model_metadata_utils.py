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
    _prepare_signature_outputs, _prepare_signature_inputs, \
    prepare_get_metadata_output


@pytest.mark.parametrize("outputs, tensor_key, dummy_shape, np_type", [
    (['tensor', 'test_tensor'], ['new_key', 'tensor_key'], (1, 1, 1),
     np.float32),
    (['tensor'], ['new_key'], (5, 3, 20), np.int32),
])
def test_prepare_signature_dummy_shape(outputs, tensor_key, dummy_shape,
                                       np_type):
    dtype_model = dtypes.as_dtype(np_type)
    model_keys = {}
    for x in range(len(outputs)):
        model_keys[tensor_key[x]] = outputs[x]

    output = _prepare_signature_outputs(names=outputs,
                                        dtype_layer=dtype_model,
                                        shape=dummy_shape,
                                        model_keys=model_keys)
    for x in range(len(outputs)):
        assert tensor_key[x] in output
        assert outputs[x] in output[tensor_key[x]].name

        shape = [d.size for d in output[tensor_key[x]].tensor_shape.dim]
        assert list(dummy_shape) == shape

        tensor_dtype = dtypes.as_dtype(output[tensor_key[x]].dtype)
        assert dtype_model == tensor_dtype


@pytest.mark.parametrize("layers, tensor_key, np_type", [
    ({'tensor': (1, 1, 1), 'test_tensor': (5, 3, 20)},
     {'new_key': 'tensor', 'client_key': 'test_tensor'}, np.float32),
    ({'tensor': (1, 1, 1)}, {'new_key': 'tensor'}, np.int32),
])
def test_prepare_signature_known_shape(layers, tensor_key, np_type):
    dtype_model = dtypes.as_dtype(np_type)
    output = _prepare_signature_inputs(layers=layers,
                                       dtype_layer=dtype_model,
                                       model_keys=tensor_key)

    for key, value in tensor_key.items():
        assert key in output
        assert value in output[key].name

        shape = [d.size for d in output[key].tensor_shape.dim]
        assert list(layers[value]) == shape

        tensor_dtype = dtypes.as_dtype(output[key].dtype)
        assert dtype_model == tensor_dtype


def test_prepare_get_metadata_output():
    inputs = {'tensor': (1, 1, 1)}
    outputs = ['tensor_output']
    model_keys = {'inputs': {'name': 'tensor'},
                  'outputs': {'output_name': 'tensor_output'}}
    output = prepare_get_metadata_output(inputs=inputs, outputs=outputs,
                                         model_keys=model_keys)

    assert "tensorflow/serving/predict" == output.method_name
