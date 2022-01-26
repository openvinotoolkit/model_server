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

from typing import NamedTuple
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.types_pb2 import DataType
import numpy as np

from ovmsclient.util.ovmsclient_export import ovmsclient_export


class TensorType(NamedTuple):
    TensorDtype: str
    TensorProtoField: str


NP_TO_TENSOR_MAP = {
    np.float16: TensorType(TensorDtype=DataType.DT_HALF, TensorProtoField="half_val"),
    np.float32: TensorType(TensorDtype=DataType.DT_FLOAT, TensorProtoField="float_val"),
    np.float64: TensorType(TensorDtype=DataType.DT_DOUBLE, TensorProtoField="double_val"),
    np.int8: TensorType(TensorDtype=DataType.DT_INT8, TensorProtoField="int_val"),
    np.int16: TensorType(TensorDtype=DataType.DT_INT16, TensorProtoField="int_val"),
    np.int32: TensorType(TensorDtype=DataType.DT_INT32, TensorProtoField="int_val"),
    np.int64: TensorType(TensorDtype=DataType.DT_INT64, TensorProtoField="int64_val"),
    np.uint8: TensorType(TensorDtype=DataType.DT_UINT8, TensorProtoField="int_val"),
    np.uint16: TensorType(TensorDtype=DataType.DT_UINT16, TensorProtoField="int_val"),
    np.uint32: TensorType(TensorDtype=DataType.DT_UINT32, TensorProtoField="uint32_val"),
    np.uint64: TensorType(TensorDtype=DataType.DT_UINT64, TensorProtoField="uint64_val"),
    np.complex64: TensorType(TensorDtype=DataType.DT_COMPLEX64, TensorProtoField="scomplex_val"),
    np.complex128: TensorType(TensorDtype=DataType.DT_COMPLEX128, TensorProtoField="dcomplex_val"),
    np.bool: TensorType(TensorDtype=DataType.DT_BOOL, TensorProtoField="bool_val"),
    np.bytes_: TensorType(TensorDtype=DataType.DT_STRING, TensorProtoField="string_val")
}


TENSOR_TO_NP_MAP = {v.TensorDtype: k for k, v in NP_TO_TENSOR_MAP.items()}


TENSOR_DTYPE_TO_PROTOFIELD = {v.TensorDtype: v.TensorProtoField for v in NP_TO_TENSOR_MAP.values()}


def _is_shape_valid(shape):
    if not isinstance(shape, (list, tuple)):
        return False
    if not all(isinstance(n, int) and n >= 0 for n in shape):
        return False
    return True


def _cast_ndarray_to_dtype(values, dtype):
    try:
        return values.astype(dtype)
    except ValueError as e_info:
        # when values has inhomogeneous shape
        raise ValueError(f'could not cast values to {dtype}. {e_info}')


def _cast_bytes_to_dtype(values, dtype):
    try:
        return np.frombuffer(values, dtype=dtype)
    except ValueError as e_info:
        # not enough bytes to fill dtype
        raise ValueError(f'could not cast bytes to {dtype}. {e_info}')


def _get_dense_dimensions(values):
    if not isinstance(values, (list, np.ndarray)):
        return []
    elif len(values) == 0:
        return [0]
    else:
        return [len(values)] + _get_dense_dimensions(values[0])


def _is_bytes_shape_valid(inferred_shape, tensor_values):
    return (len(inferred_shape) > 1 or (len(tensor_values.shape) > 1 and inferred_shape == []))


@ovmsclient_export("make_tensor_proto", grpcclient="make_tensor_proto")
def make_tensor_proto(values, dtype=None, shape=None):
    '''
    Create TensorProto object from values.

    Args:

        values: Values to put in the TensorProto.
            The accepted types are: python scalar, pythons list, numpy scalar and numpy ndarray.
            Python scalars and lists are internally converted to their numpy counterparts before further processing.
            Bytes values are placed in TensorProto string_val field.

        dtype (optional): tensor_pb2 DataType value.
            If not provided, the function will try to infer the data type from **values** argument.

        shape (optional): The list of integers defining the shape of the tensor.
            If not provided, the function will try to infer the shape from **values** argument.

    Returns:
        TensorProto object filled with **values**.

    Raises:
        TypeError:  if unsupported types are provided.
        ValueError: if arguments have inappropriate values.

    Examples:

        With python list:

        >>> data = [[1, 2, 3], [4, 5, 6]]
        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)

        With numpy array:

        >>> data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)

        With binary data:

        >>> data = bytes([1, 2, 3, 4, 5, 6])
        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)
    '''  # noqa: E501
    if shape is None:
        inferred_shape = []
    elif _is_shape_valid(shape):
        inferred_shape = list(shape)
    else:
        raise TypeError('shape type should be list or tuple with unsigned integers')

    # create numpy ndarray from values and find its dtype if not provided
    if isinstance(values, np.ndarray):
        tensor_values = values
    elif isinstance(values, list):
        dense_dimensions = _get_dense_dimensions(values)
        tensor_values = np.array(values)
        if(list(tensor_values.shape) != dense_dimensions):
            raise ValueError(f'argument must be a dense tensor: {values} - got shape '
                             f'{list(tensor_values.shape)}, but wanted {dense_dimensions}')
    elif np.isscalar(values):
        tensor_values = np.array([values])
    else:
        raise TypeError("values type should be (list, np.ndarray, scalar), but is "
                        f"{type(values).__name__}")

    if not isinstance(values, np.ndarray):
        # python/numpy default float type is float64. We prefer float32 instead.
        if (tensor_values.dtype == np.float64) and dtype is None:
            tensor_values = tensor_values.astype(np.float32)
        # python/numpy default int type is int64. We prefer int32 instead.
        elif (tensor_values.dtype == np.int64) and dtype is None:
            downcasted_array = tensor_values.astype(np.int32)
            # Do not down cast if it leads to precision loss.
            if np.array_equal(downcasted_array, tensor_values):
                tensor_values = downcasted_array

    if dtype is None:
        tensor_type = NP_TO_TENSOR_MAP.get(tensor_values.dtype.type)
        if tensor_type is not None:
            dtype = tensor_type.TensorDtype
        else:
            raise TypeError("provided values type is not valid")
    else:
        np_dtype = TENSOR_TO_NP_MAP.get(dtype)
        if np_dtype is None:
            raise TypeError(f"{dtype} is not valid dtype value")

        # values are binary, but dtype was not DT_STRING
        if tensor_values.dtype.type == np.bytes_ and dtype != DataType.DT_STRING:
            tensor_values = _cast_bytes_to_dtype(tensor_values.tobytes(), dtype=np_dtype)
        else:
            tensor_values = _cast_ndarray_to_dtype(tensor_values, np_dtype)

    if dtype == DataType.DT_STRING and _is_bytes_shape_valid(inferred_shape, tensor_values):
        raise ValueError("bytes values with dtype DT_STRING must be in shape [N]")
    elif inferred_shape == []:
        inferred_shape = list(tensor_values.shape)
    elif inferred_shape != list(tensor_values.shape):
        tensor_values = tensor_values.reshape(inferred_shape)

    dims = []
    for d in inferred_shape:
        dims.append(TensorShapeProto.Dim(size=d))
    tensor_shape = TensorShapeProto(dim=dims)
    if dtype == DataType.DT_STRING:
        tensor_proto = TensorProto(dtype=dtype, tensor_shape=tensor_shape,
                                   string_val=tensor_values.tolist())
    elif np.isscalar(values):
        # picking the right field of TensorProto to put data in
        tensor_proto_args = {
            "dtype": dtype,
            "tensor_shape": tensor_shape,
            TENSOR_DTYPE_TO_PROTOFIELD.get(dtype): tensor_values
        }
        tensor_proto = TensorProto(**tensor_proto_args)
    else:
        tensor_proto = TensorProto(dtype=dtype, tensor_shape=tensor_shape,
                                   tensor_content=tensor_values.tobytes())
    return tensor_proto


def make_ndarray(tensor_proto):
    '''
    Create numpy ndarray from tensor_proto.

    Args:
        tensor_proto: TensorProto object.

    Returns:
        Numpy ndarray with tensor contents.

    Raises:
        TypeError:  if unsupported type is provided.

    Examples:
        Create TensorProto with make_tensor_proto and convert it back to numpy array with make_ndarray:

        >>> data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        >>> print(data)

        >>> tensor_proto = make_tensor_proto(data)
        >>> print(tensor_proto)

        >>> output = make_ndarray(tensor_proto)
        >>> print(output)
    '''  # noqa: E501
    shape = [d.size for d in tensor_proto.tensor_shape.dim]
    num_elements = np.prod(shape, dtype=np.int64)
    np_dtype = TENSOR_TO_NP_MAP.get(tensor_proto.dtype)

    if tensor_proto.tensor_content:
        return (np.frombuffer(tensor_proto.tensor_content,
                              dtype=np_dtype).copy().reshape(shape))

    if np_dtype == np.bytes_:
        values = list(tensor_proto.string_val)
        padding = num_elements - len(values)
        if padding > 0:
            last = values[-1] if values else ""
            values.extend([last] * padding)
        return np.array(values, dtype=np_dtype).reshape(shape)
    elif np_dtype == np.float16:
        values = np.fromiter(tensor_proto.half_val, dtype=np.uint16)
        values.dtype = np_dtype
    elif np_dtype == np.float32:
        values = np.fromiter(tensor_proto.float_val, dtype=np_dtype)
    elif np_dtype == np.float64:
        values = np.fromiter(tensor_proto.double_val, dtype=np_dtype)
    elif np_dtype in [np.int32, np.uint8, np.uint16, np.int16, np.int8]:
        values = np.fromiter(tensor_proto.int_val, dtype=np_dtype)
    elif np_dtype == np.int64:
        values = np.fromiter(tensor_proto.int64_val, dtype=np_dtype)
    elif np_dtype == np.uint32:
        values = np.fromiter(tensor_proto.uint32_val, dtype=np_dtype)
    elif np_dtype == np.uint64:
        values = np.fromiter(tensor_proto.uint64_val, dtype=np_dtype)
    elif np_dtype == np.complex64:
        it = iter(tensor_proto.scomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=np_dtype)
    elif np_dtype == np.complex128:
        it = iter(tensor_proto.dcomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=np_dtype)
    elif np_dtype == np.bool:
        values = np.fromiter(tensor_proto.bool_val, dtype=np_dtype)
    else:
        raise TypeError("Unsupported tensor type: %s" % tensor_proto.dtype)

    if values.size == 0:
        return np.zeros(shape, np_dtype)

    if values.size != num_elements:
        values = np.pad(values, (0, num_elements - values.size), "edge")

    return values.reshape(shape)
