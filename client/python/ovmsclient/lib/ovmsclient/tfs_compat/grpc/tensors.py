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
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.tensor_pb2 import TensorProto
from ovmsclient.tfs_compat.protos.tensorflow.core.framework.types_pb2 import DataType
import numpy as np

from ovmsclient.util.ovmsclient_export import ovmsclient_export


class TensorType(NamedTuple):
    NumpyPrimaryType: np.dtype.type
    TensorDtype: str
    TensorProtoField: str


NP_TO_TENSOR_MAP = {
    np.float16: TensorType(NumpyPrimaryType=np.float16, TensorDtype=DataType.DT_HALF,
                           TensorProtoField="half_val"),
    np.float32: TensorType(NumpyPrimaryType=np.float32, TensorDtype=DataType.DT_FLOAT,
                           TensorProtoField="float_val"),
    np.float64: TensorType(NumpyPrimaryType=np.float64, TensorDtype=DataType.DT_DOUBLE,
                           TensorProtoField="double_val"),
    np.int8: TensorType(NumpyPrimaryType=np.int8, TensorDtype=DataType.DT_INT8,
                        TensorProtoField="int_val"),
    np.int16: TensorType(NumpyPrimaryType=np.int16, TensorDtype=DataType.DT_INT16,
                         TensorProtoField="int_val"),
    np.int32: TensorType(NumpyPrimaryType=np.int32, TensorDtype=DataType.DT_INT32,
                         TensorProtoField="int_val"),
    np.int64: TensorType(NumpyPrimaryType=np.int64, TensorDtype=DataType.DT_INT64,
                         TensorProtoField="int64_val"),
    np.uint8: TensorType(NumpyPrimaryType=np.uint8, TensorDtype=DataType.DT_UINT8,
                         TensorProtoField="int_val"),
    np.uint16: TensorType(NumpyPrimaryType=np.uint16, TensorDtype=DataType.DT_UINT16,
                          TensorProtoField="int_val"),
    np.uint32: TensorType(NumpyPrimaryType=np.uint32, TensorDtype=DataType.DT_UINT32,
                          TensorProtoField="uint32_val"),
    np.uint64: TensorType(NumpyPrimaryType=np.uint64, TensorDtype=DataType.DT_UINT64,
                          TensorProtoField="uint64_val"),
    np.complex64: TensorType(NumpyPrimaryType=np.complex64, TensorDtype=DataType.DT_COMPLEX64,
                             TensorProtoField="scomplex_val"),
    np.complex128: TensorType(NumpyPrimaryType=np.complex128, TensorDtype=DataType.DT_COMPLEX128,
                              TensorProtoField="dcomplex_val"),
    # Standard Python bool and np.bool_ replace deprecated np.bool type
    bool: TensorType(NumpyPrimaryType=np.bool_, TensorDtype=DataType.DT_BOOL,
                     TensorProtoField="bool_val"),
    np.bool_: TensorType(NumpyPrimaryType=np.bool_, TensorDtype=DataType.DT_BOOL,
                         TensorProtoField="bool_val"),
    np.str_: TensorType(NumpyPrimaryType=np.bytes_, TensorDtype=DataType.DT_STRING,
                        TensorProtoField="string_val"),
    np.bytes_: TensorType(NumpyPrimaryType=np.bytes_, TensorDtype=DataType.DT_STRING,
                          TensorProtoField="string_val"),
}


TENSOR_TO_NP_MAP = {v.TensorDtype: v.NumpyPrimaryType for v in NP_TO_TENSOR_MAP.values()}


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


# Functioned used with numpy.vectorize()
def _encode_string_to_bytes(tensor_value, encoding="UTF-8"):
    return tensor_value.encode(encoding)


_encode_strings_to_bytes = np.vectorize(_encode_string_to_bytes)


def _check_if_array_homogeneous(tensor_values):
    #  Exception details match numpy ValueError thrown on attempt to create inhomogeneous
    #  numpy array in versions >=1.24. This function has been created to provide
    #  the same user experience for both pre and post 1.24 numpy versions.
    if tensor_values.dtype.type is np.object_ or tensor_values.dtype.type is object:
        raise ValueError('setting an array element with a sequence. The requested array has '
                         f'an inhomogeneous shape after {len(tensor_values.shape)} dimensions. '
                         f'The detected shape was {tensor_values.shape} + inhomogeneous part.')


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
        tensor_values = np.array(values)
        _check_if_array_homogeneous(tensor_values)
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

    if inferred_shape == []:
        inferred_shape = list(tensor_values.shape)
    elif inferred_shape != list(tensor_values.shape):
        tensor_values = tensor_values.reshape(inferred_shape)

    # For strings or binary image inputs flatten array to 1-D
    if dtype == DataType.DT_STRING:
        tensor_values = np.ravel(tensor_values)
        # Encode strings
        if tensor_values.dtype.type == np.str_:
            tensor_values = _encode_strings_to_bytes(tensor_values)

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
    elif np_dtype == np.bool_:
        values = np.fromiter(tensor_proto.bool_val, dtype=np_dtype)
    else:
        raise TypeError("Unsupported tensor type: %s" % tensor_proto.dtype)

    if values.size == 0:
        return np.zeros(shape, np_dtype)

    if values.size != num_elements:
        values = np.pad(values, (0, num_elements - values.size), "edge")

    return values.reshape(shape)
