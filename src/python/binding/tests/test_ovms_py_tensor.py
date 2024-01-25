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


import numpy as np
from pyovms import Tensor
import pytest

NUMPY_NUMERICAL_TYPES = [np.bool_, np.uint8, np.uint16, np.uint32, np.uint64,
                        np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]

@pytest.mark.parametrize("array_dtype", NUMPY_NUMERICAL_TYPES)
def test_creating_from_numpy_numerical_array(array_dtype):
    npy_arr = np.array(np.ones((1,3,300,300), dtype=array_dtype))
    ovms_py_tensor = Tensor("input", npy_arr)
    assert ovms_py_tensor.name == "input"
    assert npy_arr.shape == ovms_py_tensor.shape
    assert npy_arr.data.shape == ovms_py_tensor.data.shape
    assert npy_arr.data.format == ovms_py_tensor.data.format
    assert npy_arr.data.itemsize == ovms_py_tensor.data.itemsize
    assert npy_arr.data.strides == ovms_py_tensor.data.strides
    assert npy_arr.data == ovms_py_tensor.data

@pytest.mark.parametrize("data", ["single_string", ["batch", "of", "strings"]])
def test_creating_from_numpy_string_array(data):
    npy_arr = np.array(data)
    ovms_py_tensor = Tensor("input", npy_arr)
    assert ovms_py_tensor.name == "input"
    assert npy_arr.shape == ovms_py_tensor.shape
    assert npy_arr.data.shape == ovms_py_tensor.data.shape
    assert npy_arr.data.format == ovms_py_tensor.data.format
    assert npy_arr.data.itemsize == ovms_py_tensor.data.itemsize
    assert npy_arr.data.strides == ovms_py_tensor.data.strides
    assert npy_arr.data.tobytes() == ovms_py_tensor.data.tobytes()

def test_creating_from_python_bytes():
    data = "some string encoded with default UTF-8 encoding".encode()
    ovms_py_tensor = Tensor("input", data)
    assert ovms_py_tensor.name == "input"
    assert ovms_py_tensor.shape == (len(data),)
    assert ovms_py_tensor.data.ndim == 1
    assert ovms_py_tensor.data.shape == (len(data),)
    assert ovms_py_tensor.data.format == "B" # UINT8
    assert ovms_py_tensor.datatype == "UINT8"
    assert ovms_py_tensor.data.itemsize == 1
    assert ovms_py_tensor.data.strides == (1,)
    assert ovms_py_tensor.data.tobytes() == data
