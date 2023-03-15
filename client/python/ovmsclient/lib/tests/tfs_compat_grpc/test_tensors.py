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
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_pb2 import TensorProto
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.types_pb2 import DataType
from ovmsclient.tfs_compat.grpc.tensors import TENSOR_TO_NP_MAP, make_ndarray, make_tensor_proto
import numpy as np

shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)])


def test_make_ndarray_valid_int():
    tensor_proto = TensorProto(tensor_shape=shape, dtype=DataType.DT_INT8,
                               tensor_content=bytes([1, 2, 3]))
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [1, 2, 3]
    assert array.dtype == np.int8


def test_make_ndarray_valid_float():
    content = [1.0, 2.0, 3.0]
    np_content = np.array(content, dtype=np.float32)
    tensor_proto = TensorProto(tensor_shape=shape, dtype=DataType.DT_FLOAT,
                               tensor_content=np_content.tobytes())
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [1.0, 2.0, 3.0]
    assert array.dtype == np.float32


def test_make_ndarray_valid_3_dims_shape():
    content = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    np_content = np.array(content, dtype=np.float32)
    _shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=1), TensorShapeProto.Dim(size=2),
                              TensorShapeProto.Dim(size=3)])
    tensor_proto = TensorProto(tensor_shape=_shape, dtype=DataType.DT_FLOAT,
                               tensor_content=np_content.tobytes())
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]
    assert array.dtype == np.float32


dtype_4_bytes = 'S4'


def test_make_ndarray_valid_string():
    hex_string = "11111111"
    tensor_proto = TensorProto(tensor_shape=shape, dtype=DataType.DT_STRING,
                               string_val=[bytes.fromhex(hex_string)])
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [b'\x11\x11\x11\x11', b'\x11\x11\x11\x11', b'\x11\x11\x11\x11']
    assert array.dtype == dtype_4_bytes


def test_make_ndarray_valid_no_content():
    tensor_proto = TensorProto(tensor_shape=shape, dtype=DataType.DT_INT8)
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [0, 0, 0]
    assert array.dtype == np.int8


dtype_1_byte = 'S1'


def test_make_ndarray_valid_no_content_string():
    tensor_proto = TensorProto(tensor_shape=shape, dtype=DataType.DT_STRING)
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [b'', b'', b'']
    assert array.dtype == dtype_1_byte


def test_make_ndarray_valid_no_content_string_with_other_type_content():
    tensor_proto = TensorProto(tensor_shape=shape, dtype=DataType.DT_STRING)
    array = make_ndarray(tensor_proto)
    assert array.tolist() == [b'', b'', b'']
    assert array.dtype == dtype_1_byte


def test_make_ndarray_invalid_type():
    tensor_proto = TensorProto(tensor_shape=shape)
    tensor_proto.dtype = 0
    with pytest.raises(TypeError) as exception_info:
        make_ndarray(tensor_proto)
    exception = exception_info.value
    assert str(exception) == "Unsupported tensor type: 0"


def test_make_ndarray_invalid_no_shape():
    tensor_proto = TensorProto(dtype=DataType.DT_INT8, tensor_content=bytes([1, 2, 3]))
    with pytest.raises(ValueError) as exception_info:
        make_ndarray(tensor_proto)
    exception = exception_info.value
    assert str(exception) == "cannot reshape array of size 3 into shape ()"


def test_make_ndarray_invalid_shape_does_not_match():
    tensor_proto = TensorProto(tensor_shape=shape, dtype=DataType.DT_INT8,
                               tensor_content=bytes([1, 2, 3, 4]))
    with pytest.raises(ValueError) as exception_info:
        make_ndarray(tensor_proto)
    exception = exception_info.value
    assert str(exception) == "cannot reshape array of size 4 into shape (3,)"


@pytest.mark.parametrize("params, expected_shape, expected_dtype", [
    ({"values": [1, 2, 3]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT32
     ),
    ({"values": [1, 2, 3], "shape": [3]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT32
     ),
    ({"values": [1, 2, 3], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT8
     ),
    ({"values": [1, 2, 3], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_FLOAT
     ),
    ({"values": [1, 2, 3], "shape": [3], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT8
     ),
    ({"values": np.array([1, 2, 3])},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT64
     ),
    ({"values": np.array([1, 2, 3]), "shape": [3]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT64
     ),
    ({"values": np.array([1, 2, 3]), "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT8
     ),
    ({"values": np.array([1, 2, 3]), "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_FLOAT
     ),
    ({"values": np.array([1, 2, 3]), "shape": [3], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT8
     ),
    ({"values": np.array([1, 2, 3], dtype=np.int64), "shape": [3], "dtype": DataType.DT_INT16},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT16
     ),
    ({"values": np.array([1, 2, 3], dtype=np.int16), "shape": [3], "dtype": DataType.DT_INT64},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT64
     ),
])
def test_make_tensor_proto_valid_int(params, expected_shape, expected_dtype):
    tensor_proto = make_tensor_proto(**params)
    np_dtype = TENSOR_TO_NP_MAP.get(expected_dtype)
    np_values = np.array(params["values"], dtype=np_dtype)

    assert tensor_proto.tensor_content == np_values.tobytes()
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


@pytest.mark.parametrize("params, expected_shape, expected_dtype", [
    ({"values": []},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=0)]), DataType.DT_FLOAT
     ),
    ({"values": [], "shape": []},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=0)]), DataType.DT_FLOAT
     ),
    ({"values": [], "shape": [0]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=0)]), DataType.DT_FLOAT
     ),
    ({"values": [], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=0)]), DataType.DT_INT8
     ),
    ({"values": [], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=0)]), DataType.DT_FLOAT
     ),
    ({"values": [], "shape": [], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=0)]), DataType.DT_INT8
     ),
    ({"values": [], "shape": [0], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=0)]), DataType.DT_INT8
     ),
])
def test_make_tensor_proto_valid_empty_list(params, expected_shape, expected_dtype):
    tensor_proto = make_tensor_proto(**params)
    np_dtype = TENSOR_TO_NP_MAP.get(expected_dtype)
    np_values = np.array(params["values"], dtype=np_dtype)

    assert tensor_proto.tensor_content == np_values.tobytes()
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


@pytest.mark.parametrize("params, expected_shape, expected_dtype", [
    ({"values": [[], [], []]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=0)]),
     DataType.DT_FLOAT
     ),
    ({"values": [[], [], []], "shape": [3, 0]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=0)]),
     DataType.DT_FLOAT
     ),
    ({"values": [[], [], []], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=0)]),
     DataType.DT_INT8
     ),
    ({"values": [[], [], []], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=0)]),
     DataType.DT_FLOAT
     ),
    ({"values": [[], [], []], "shape": [3, 0], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=0)]),
     DataType.DT_INT8
     ),
])
def test_make_tensor_proto_valid_empty_list_of_empty_lists(params, expected_shape, expected_dtype):
    tensor_proto = make_tensor_proto(**params)
    np_dtype = TENSOR_TO_NP_MAP.get(expected_dtype)
    np_values = np.array(params["values"], dtype=np_dtype)

    assert tensor_proto.tensor_content == np_values.tobytes()
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


@pytest.mark.parametrize("params, expected_shape, expected_dtype", [
    ({"values": [1.0, 2.0, 3.0]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_FLOAT
     ),
    ({"values": [1.0, 2.0, 3.0], "shape": [3]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_FLOAT
     ),
    ({"values": [1.0, 2.0, 3.0], "dtype": DataType.DT_INT16},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT16
     ),
    ({"values": [1.0, 2.0, 3.0], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_FLOAT
     ),
    ({"values": [1.0, 2.0, 3.0], "shape": [3], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_INT8
     ),
    ({"values": [1.0, 2.0, 3.0], "shape": [3], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]), DataType.DT_FLOAT
     ),
])
def test_make_tensor_proto_valid_float(params, expected_shape, expected_dtype):
    tensor_proto = make_tensor_proto(**params)
    np_dtype = TENSOR_TO_NP_MAP.get(expected_dtype)
    np_values = np.array(params["values"], dtype=np_dtype)

    assert tensor_proto.tensor_content == np_values.tobytes()
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


@pytest.mark.parametrize("params, expected_shape, expected_dtype, expected_field", [
    ({"values": 5.0},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": 5.0, "shape": [1]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": 5.0, "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_INT32,
     "int_val"
     ),
    ({"values": 5.0, "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": 5.0, "shape": [1], "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_INT32,
     "int_val"
     ),
    ({"values": 5.0, "shape": [1], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": np.float32(5.0)},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": np.float32(5.0), "shape": [1]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": np.float32(5.0), "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_INT32,
     "int_val"
     ),
    ({"values": np.float32(5.0), "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": np.float32(5.0), "shape": [1], "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_INT32,
     "int_val"
     ),
    ({"values": np.float32(5.0), "shape": [1], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_FLOAT,
     "float_val"
     ),
    ({"values": np.float32(5.0), "shape": [1], "dtype": DataType.DT_DOUBLE},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_DOUBLE,
     "double_val"
     ),
])
def test_make_tensor_proto_valid_scalar(params, expected_shape, expected_dtype, expected_field):
    tensor_proto = make_tensor_proto(**params)
    np_dtype = TENSOR_TO_NP_MAP.get(expected_dtype)
    np_values = np.array(params["values"], dtype=np_dtype)

    assert tensor_proto.__getattribute__(expected_field) == np_values
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


@pytest.mark.parametrize("params, expected_shape, expected_dtype, expected_field", [
    ({"values": bytes([0x13, 0x00, 0x00, 0x00, 0x08])},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_STRING,
     "string_val"
     ),
    ({"values": bytes([0x13, 0x00, 0x00, 0x00, 0x08]), "shape": [1]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_STRING,
     "string_val"
     ),
    ({"values": bytes([0x13, 0x00, 0x00, 0x00, 0x08]), "dtype": DataType.DT_STRING},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_STRING,
     "string_val"
     ),
    ({"values": bytes([0x13, 0x00, 0x00, 0x00, 0x08]), "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=5)]), DataType.DT_INT8,
     "int_val"
     ),
    ({"values": bytes([0x13, 0x00, 0x00, 0x00, 0x08]), "shape": [5], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=5)]), DataType.DT_INT8,
     "int_val"
     ),
    ({"values": bytes([0x13, 0x00, 0x00, 0x00, 0x08]), "shape": [1], "dtype": DataType.DT_STRING},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]), DataType.DT_STRING,
     "string_val"
     ),
    ({"values": [[bytes([0x13, 0x00, 0x00, 0x00, 0x08]), bytes([0x13, 0x00, 0x00, 0x00, 0x08])]],
      "dtype": DataType.DT_STRING},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1), TensorShapeProto.Dim(size=2)]),
     DataType.DT_STRING,
     "string_val"
     ),
])
def test_make_tensor_proto_valid_binary(params, expected_shape, expected_dtype, expected_field):
    tensor_proto = make_tensor_proto(**params)

    if expected_field == "string_val":
        if type(params["values"]) is not list:
            assert tensor_proto.__getattribute__(expected_field) == [params["values"]]
        else:
            assert (tensor_proto.__getattribute__(expected_field)
                    == np.ravel(params["values"]).tolist())
    else:
        assert (tensor_proto.__getattribute__(expected_field)
                == np.frombuffer(params["values"],
                                 dtype=TENSOR_TO_NP_MAP.get(expected_dtype)).tolist())
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


@pytest.mark.parametrize("params, expected_shape", [
    ({"values": "string", "dtype": DataType.DT_STRING},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)])
     ),
    ({"values": ["list", "of", "strings"], "shape": [3], "dtype": DataType.DT_STRING},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)])
     ),
    ({"values": [["nested", "list", "of", "strings"]]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=1), TensorShapeProto.Dim(size=4)])
     ),
    # Upon numpy array creation it will be casted to numpy.str_ data type
    ({"values": [1, 2, "three"]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)])
     ),
    ({"values": [[1, 2], [3, "four"]]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=2)])
     ),
])
def test_make_tensor_proto_valid_string(params, expected_shape):
    tensor_proto = make_tensor_proto(**params)
    assert tensor_proto.string_val == np.ravel(params["values"]).astype(np.bytes_).tolist()
    assert tensor_proto.dtype == DataType.DT_STRING
    assert tensor_proto.tensor_shape == expected_shape


def test_make_tensor_proto_valid_string_to_float_dtype():
    values = bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])
    tensor_proto = make_tensor_proto(values=values, shape=[3], dtype=DataType.DT_INT16)
    np_values = np.frombuffer(np.array(values).tobytes(), dtype=np.int16).tolist()

    assert tensor_proto.int_val == np_values
    assert tensor_proto.dtype == DataType.DT_INT16
    assert tensor_proto.tensor_shape == TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)])


@pytest.mark.parametrize("params, expected_shape, expected_dtype, expected_field", [
    ({"values": [bytes([0x13, 0x00, 0x00, 0x00, 0x08]), bytes([0x13, 0x00, 0x00, 0x00, 0x08])]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2)]), DataType.DT_STRING,
     "string_val"
     ),
    ({"values": [bytes([0x13, 0x00, 0x00, 0x00, 0x08]), bytes([0x13, 0x00, 0x00, 0x00, 0x08])],
      "shape": [2]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2)]), DataType.DT_STRING,
     "string_val"
     ),
    ({"values": [bytes([0x13, 0x00, 0x00, 0x00, 0x08]), bytes([0x13, 0x00, 0x00, 0x00, 0x08])],
      "dtype": DataType.DT_STRING},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2)]), DataType.DT_STRING,
     "string_val"
     ),
    ({"values": [bytes([0x13, 0x00, 0x00, 0x00, 0x08]), bytes([0x13, 0x00, 0x00, 0x00, 0x08])],
      "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=10)]), DataType.DT_INT8,
     "tensor_content"
     ),
    ({"values": [bytes([0x13, 0x00, 0x00, 0x00, 0x08]), bytes([0x13, 0x00, 0x00, 0x00, 0x08])],
      "shape": [10], "dtype": DataType.DT_INT8},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=10)]), DataType.DT_INT8,
     "tensor_content"
     ),
    ({"values": [bytes([0x13, 0x00, 0x00, 0x00, 0x08]), bytes([0x13, 0x00, 0x00, 0x00, 0x08])],
      "shape": [2], "dtype": DataType.DT_STRING},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2)]), DataType.DT_STRING,
     "string_val"
     ),
])
def test_make_tensor_proto_valid_string_batch_size_2(params, expected_shape, expected_dtype,
                                                     expected_field):
    tensor_proto = make_tensor_proto(**params)
    np_values = np.array(params["values"])

    if expected_field == "string_val":
        assert tensor_proto.__getattribute__(expected_field) == params["values"]
    elif expected_field == "tensor_content":
        assert tensor_proto.__getattribute__(expected_field) == np_values.tobytes()
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


@pytest.mark.parametrize("params, expected_shape, expected_dtype", [
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_FLOAT,
     ),
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "shape": [2, 3]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_FLOAT,
     ),
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "shape": [6]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=6)]), DataType.DT_FLOAT,
     ),
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_INT32,
     ),
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_FLOAT,
     ),
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "shape": [2, 3], "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_INT32,
     ),
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "shape": [2, 3], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_FLOAT,
     ),
    ({"values": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "shape": [6], "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=6)]), DataType.DT_FLOAT,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_DOUBLE,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), "shape": [2, 3]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_DOUBLE,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), "shape": [6]},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=6)]), DataType.DT_DOUBLE,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_INT32,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_FLOAT,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), "shape": [2, 3],
      "dtype": DataType.DT_INT32},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_INT32,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), "shape": [2, 3],
      "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_FLOAT,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), "shape": [6],
      "dtype": DataType.DT_FLOAT},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=6)]), DataType.DT_FLOAT,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32),
      "dtype": DataType.DT_HALF},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_HALF,
     ),
    ({"values": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float16),
      "dtype": DataType.DT_DOUBLE},
     TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)]),
     DataType.DT_DOUBLE,
     ),
])
def test_make_tensor_proto_valid_2_dims_shape(params, expected_shape, expected_dtype):
    tensor_proto = make_tensor_proto(**params)
    np_dtype = TENSOR_TO_NP_MAP.get(expected_dtype)
    np_values = np.array(params["values"], dtype=np_dtype)

    assert tensor_proto.tensor_content == np_values.tobytes()
    assert tensor_proto.dtype == expected_dtype
    assert tensor_proto.tensor_shape == expected_shape


def test_make_tensor_proto_valid_make_ndarray_valid():
    values = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    np_values = np.array(values, dtype=np.float32)
    _shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=2), TensorShapeProto.Dim(size=3)])
    tensor_proto = TensorProto(tensor_shape=_shape, dtype=DataType.DT_FLOAT,
                               tensor_content=np_values.tobytes())
    array = make_ndarray(tensor_proto)
    tensor_proto = make_tensor_proto(values=array, dtype=DataType.DT_FLOAT)
    assert tensor_proto.tensor_content == np_values.tobytes()
    assert tensor_proto.dtype == DataType.DT_FLOAT
    assert tensor_proto.tensor_shape == _shape


def test_make_tensor_proto_valid_string_reshape():
    values = bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])
    tensor_proto = make_tensor_proto(values=[[values, values], [values, values]],
                                     shape=[4], dtype=DataType.DT_STRING)
    assert all(val == np.array(values) for val in tensor_proto.string_val)
    assert tensor_proto.dtype == DataType.DT_STRING
    assert tensor_proto.tensor_shape == TensorShapeProto(dim=[TensorShapeProto.Dim(size=4)])


def test_make_tensor_proto_valid_2D_array_to_bytes():
    values = [[1, 2, 3], [4, 5, 6]]
    tensor_proto = make_tensor_proto(values=values, shape=[6], dtype=DataType.DT_STRING)
    assert tensor_proto.string_val == np.array(values, dtype=np.bytes_).reshape([6]).tolist()
    assert tensor_proto.dtype == DataType.DT_STRING
    assert tensor_proto.tensor_shape == TensorShapeProto(dim=[TensorShapeProto.Dim(size=6)])


def test_make_tensor_proto_invalid_shape_element_values():
    values = [1, 2, 3]
    with pytest.raises(TypeError) as exception_info:
        make_tensor_proto(values=values, shape=[-3], dtype=None)
    exception = exception_info.value
    assert str(exception) == "shape type should be list or tuple with unsigned integers"


@pytest.mark.causes_deprecation_warning(triggered_by="numpy<1.24")
def test_make_tensor_proto_invalid_dimensions():
    values = [[1.0, 2.0], [1.0, 2.0, 3.0]]
    with pytest.raises(ValueError) as exception_info:
        make_tensor_proto(values=values, shape=[2, 3], dtype=DataType.DT_FLOAT)
    exception = exception_info.value
    assert str(exception) == ("setting an array element with a sequence. "
                              "The requested array has an inhomogeneous shape after 1 dimensions. "
                              "The detected shape was (2,) + inhomogeneous part.")


def test_make_tensor_proto_invalid_string_to_float_dtype():
    values = bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])
    with pytest.raises(ValueError) as exception_info:
        make_tensor_proto(values=values, shape=None, dtype=DataType.DT_FLOAT)
    exception = exception_info.value
    assert str(exception) == ("could not cast bytes to <class 'numpy.float32'>. "
                              "buffer size must be a multiple of element size")


@pytest.mark.causes_deprecation_warning(triggered_by="numpy<1.24")
def test_make_tensor_proto_invalid_string_dimensions():
    values = bytes([0x13, 0x00])
    with pytest.raises(ValueError) as exception_info:
        make_tensor_proto(values=[[values, values, values], [values, values]],
                          shape=None, dtype=DataType.DT_STRING)
    exception = exception_info.value
    assert str(exception) == ("setting an array element with a sequence. "
                              "The requested array has an inhomogeneous shape after 1 dimensions. "
                              "The detected shape was (2,) + inhomogeneous part.")


@pytest.mark.causes_deprecation_warning(triggered_by="numpy<1.24")
def test_make_tensor_proto_invalid_dimensions_2():
    values = [[(1, 2, 3)], [(1, 2)], [(1, 2, 3)]]
    with pytest.raises(ValueError) as exception_info:
        make_tensor_proto(values=values, shape=[2, 3], dtype=DataType.DT_FLOAT)
    exception = exception_info.value
    assert str(exception) == ("setting an array element with a sequence. "
                              "The requested array has an inhomogeneous shape after 2 dimensions. "
                              "The detected shape was (3, 1) + inhomogeneous part.")


@pytest.mark.causes_deprecation_warning(triggered_by="numpy<1.24")
def test_make_tensor_proto_invalid_dimensions_no_shape_provided():
    values = [[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2, 3], [4]]]
    with pytest.raises(ValueError) as exception_info:
        make_tensor_proto(values=values, shape=None, dtype=DataType.DT_INT8)
    exception = exception_info.value
    assert str(exception) == ("setting an array element with a sequence. "
                              "The requested array has an inhomogeneous shape after 2 dimensions. "
                              "The detected shape was (3, 2) + inhomogeneous part.")


def test_make_tensor_proto_invalid_shape_type():
    values = 5.0
    with pytest.raises(TypeError) as exception_info:
        make_tensor_proto(values=values, shape=0, dtype=DataType.DT_FLOAT)
    exception = exception_info.value
    assert str(exception) == "shape type should be list or tuple with unsigned integers"


def test_make_tensor_proto_invalid_values_dtype():
    values = [np.float128(2.5)]
    with pytest.raises(TypeError) as exception_info:
        make_tensor_proto(values=values, shape=None, dtype=None)
    exception = exception_info.value
    assert str(exception) == "provided values type is not valid"


def test_make_tensor_proto_invalid_dtype_value():
    values = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    with pytest.raises(TypeError) as exception_info:
        make_tensor_proto(values=values, shape=None, dtype=int)
    exception = exception_info.value
    assert str(exception) == "<class 'int'> is not valid dtype value"


def test_make_tensor_proto_invalid_values_type():
    values = (1, 2, 3)
    with pytest.raises(TypeError) as exception_info:
        make_tensor_proto(values=values, shape=None, dtype=None)
    exception = exception_info.value
    assert str(exception) == "values type should be (list, np.ndarray, scalar), but is tuple"


def test_make_tensor_proto_invalid_int_reshape():
    values = [1, 2, 3]
    with pytest.raises(ValueError) as exception_info:
        make_tensor_proto(values=values, shape=[4], dtype=DataType.DT_INT8)
    exception = exception_info.value
    assert str(exception) == "cannot reshape array of size 3 into shape (4,)"


def test_make_tensor_proto_invalid_empty_list_of_empty_lists_reshape():
    values = [[], [], []]
    with pytest.raises(ValueError) as exception_info:
        make_tensor_proto(values=values, shape=[4, 2], dtype=DataType.DT_INT8)
    exception = exception_info.value
    assert str(exception) == "cannot reshape array of size 0 into shape (4,2)"
