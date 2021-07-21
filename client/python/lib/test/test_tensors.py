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
from tensorflow.core.framework.tensor_shape_pb2 import *
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.types_pb2 import DataType
from ovmsclient.tfs_compat.grpc.tensors import make_ndarray
import numpy as np

shape = TensorShapeProto(dim = [TensorShapeProto.Dim(size=3)])

def test_make_ndarray_valid_int():
    tensor_proto = TensorProto(tensor_shape=shape, dtype = DataType.DT_INT8, tensor_content = bytes([1,2,3]))
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [1, 2, 3]
    assert array.dtype == np.int8

def test_make_ndarray_valid_float():
    content = [1.0, 2.0, 3.0]
    np_content = np.array(content, dtype = np.float32)
    tensor_proto = TensorProto(tensor_shape=shape, dtype = DataType.DT_FLOAT, tensor_content = np_content.tobytes())
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [1.0, 2.0, 3.0]
    assert array.dtype == np.float32

def test_make_ndarray_valid_3_dims_shape():
    content = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    np_content = np.array(content, dtype = np.float32)
    _shape = TensorShapeProto(dim = [TensorShapeProto.Dim(size=1), TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)])
    tensor_proto = TensorProto(tensor_shape=_shape, dtype = DataType.DT_FLOAT, tensor_content = np_content.tobytes())
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]
    assert array.dtype == np.float32

def test_make_ndarray_valid_string():
    hex_string = "11111111"
    tensor_proto = TensorProto(tensor_shape=shape, dtype = DataType.DT_STRING, string_val = [bytes.fromhex(hex_string)])
    array = make_ndarray(tensor_proto)
    assert array.tolist() == ['\x11\x11\x11\x11', '\x11\x11\x11\x11', '\x11\x11\x11\x11']
    assert array.dtype == '<U4'

def test_make_ndarray_valid_no_content():
    tensor_proto = TensorProto(tensor_shape=shape, dtype = DataType.DT_INT8)
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [0, 0, 0]
    assert array.dtype == np.int8

def test_make_ndarray_valid_no_content_string():
    tensor_proto = TensorProto(tensor_shape=shape, dtype = DataType.DT_STRING)
    array = make_ndarray(tensor_proto)
    assert array.tolist() == ['', '', '']
    assert array.dtype == '<U1'

def test_make_ndarray_valid_no_content_string_with_other_type_content():
    tensor_proto = TensorProto(tensor_shape=shape, dtype = DataType.DT_STRING)
    array = make_ndarray(tensor_proto)
    tensor_proto.tensor_content = bytes([1,2,3])
    assert array.tolist() == ['', '', '']
    assert array.dtype == '<U1'

def test_make_ndarray_invalid_type():
    tensor_proto = TensorProto(tensor_shape=shape)
    tensor_proto.dtype = 0
    with pytest.raises(ValueError) as exception_info:
        make_ndarray(tensor_proto)

    exception = exception_info.value
    assert str(exception) == "Tensor data type invalid"

def test_make_ndarray_invalid_no_shape():
    tensor_proto = TensorProto(dtype = DataType.DT_INT8, tensor_content = bytes([1,2,3]))
    with pytest.raises(ValueError) as exception_info:
        make_ndarray(tensor_proto)

    exception = exception_info.value
    assert str(exception) == "cannot reshape array of size 3 into shape ()"

def test_make_ndarray_invalid_shape_does_not_match():
    tensor_proto = TensorProto(tensor_shape=shape, dtype = DataType.DT_INT8, tensor_content = bytes([1,2,3,4]))
    with pytest.raises(ValueError) as exception_info:
        make_ndarray(tensor_proto)

    exception = exception_info.value
    assert str(exception) == "cannot reshape array of size 4 into shape (3,)"

def test_make_ndarray_invalid_no_type():
    tensor_proto = TensorProto(tensor_shape=shape, tensor_content = bytes([1,2,3]))
    with pytest.raises(ValueError) as exception_info:
        make_ndarray(tensor_proto)

    exception = exception_info.value
    assert str(exception) == "Tensor data type invalid"
