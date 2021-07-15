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

import pytest
import numpy as np
from ovmsclient.tfs_compat.grpc.tensors import make_ndarray
from tensorflow.core.framework.tensor_shape_pb2 import *

shape = TensorShapeProto(dim = [TensorShapeProto.Dim(size=3)])

def test_make_ndarray_valid_int():
    tensor_proto = TensorProto(tensor_shape=shape, dtype = 6, tensor_content = bytes([1,2,3]))
    array = make_ndarray(tensor_proto)
    assert str(array) == "[1 2 3]"

def test_make_ndarray_valid_float():
    hex_string = "0000803f0000004000004040"
    tensor_proto = TensorProto(tensor_shape=shape, dtype = 1, tensor_content = bytes.fromhex(hex_string))
    array = make_ndarray(tensor_proto)
    assert str(array) == "[1. 2. 3.]"

def test_make_ndarray_valid_3_dims_shape():
    hex_string = "0000803f00000040000040400000803f0000004000004040"
    _shape = TensorShapeProto(dim = [TensorShapeProto.Dim(size=1), TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)])
    tensor_proto = TensorProto(tensor_shape=_shape, dtype = 1, tensor_content = bytes.fromhex(hex_string))
    array = make_ndarray(tensor_proto)
    assert str(array) == "[[[1. 2. 3.]\n  [1. 2. 3.]]]"

def test_make_ndarray_valid_string():
    hex_string = "11111111"
    tensor_proto = TensorProto(tensor_shape=shape, dtype = 7, string_val = [bytes.fromhex(hex_string)])
    array = make_ndarray(tensor_proto)
    assert str(array) == "['\\x11\\x11\\x11\\x11' '\\x11\\x11\\x11\\x11' '\\x11\\x11\\x11\\x11']"

def test_make_ndarray_invalid_type():
    tensor_proto = TensorProto(tensor_shape=shape)
    tensor_proto.dtype = 0
    with pytest.raises(ValueError) as exception_info:
        make_ndarray(tensor_proto)

    exception = exception_info.value
    assert str(exception) == "Tensor data type invalid"
