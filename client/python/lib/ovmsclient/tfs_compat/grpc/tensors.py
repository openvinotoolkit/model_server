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

from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.types_pb2 import DataType
import numpy as np

def _as_numpy_dtype(tensor_dtype):
    if tensor_dtype == DataType.DT_INVALID:
        raise ValueError("Tensor data type invalid")
    elif tensor_dtype == DataType.DT_FLOAT:
        return np.float32
    elif tensor_dtype == DataType.DT_DOUBLE:
        return np.float64
    elif tensor_dtype == DataType.DT_INT32:
        return np.int32
    elif tensor_dtype == DataType.DT_UINT8:
        return np.uint8
    elif tensor_dtype == DataType.DT_INT16:
        return np.int16
    elif tensor_dtype == DataType.DT_INT8:
        return np.int8
    elif tensor_dtype == DataType.DT_STRING:
        return np.str
    elif tensor_dtype == DataType.DT_COMPLEX64:
        return np.complex64
    elif tensor_dtype == DataType.DT_COMPLEX128:
        return np.complex128
    elif tensor_dtype == DataType.DT_INT64:
        return np.int64
    elif tensor_dtype == DataType.DT_BOOL:
        return np.bool
    elif tensor_dtype == DataType.DT_UINT16:
        return np.uint16
    elif tensor_dtype == DataType.DT_UINT32:
        return np.uint32
    elif tensor_dtype == DataType.DT_UINT64:
        return np.uint64
    else:
        return np.uint32

def _is_array_like(value):
    return isinstance(value, list) or isinstance(value, np.ndarray)

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
    '''
    if shape is None:
        infered_shape = []
    elif _is_array_like(shape):
        infered_shape = shape
    else:
        raise TypeError(f'shape type should be list, but is {type(shape).__name__}')
    
    value = values
    while dtype is None or (shape is None and _is_array_like(value)):
        if _is_array_like(value):
            if shape is None:
                infered_shape.append(len(value))
            if len(value)>0:
                value = value[0]
            else:
                value = None
        elif isinstance(value, str):
            dtype = DataType.DT_STRING
        elif isinstance(value, np.float64):
            dtype = DataType.DT_DOUBLE
        elif isinstance(value, np.float32):
            dtype = DataType.DT_FLOAT
        elif isinstance(value, float):
            dtype = DataType.DT_DOUBLE
        elif isinstance(value, np.uint8):
            dtype = DataType.DT_UINT8
        elif isinstance(value, np.uint16):
            dtype = DataType.DT_UINT16
        elif isinstance(value, np.uint32):
            dtype = DataType.DT_UINT32
        elif isinstance(value, np.uint64):
            dtype = DataType.DT_UINT64
        elif isinstance(value, np.int8):
            dtype = DataType.DT_INT8
        elif isinstance(value, np.int16):
            dtype = DataType.DT_INT16
        elif isinstance(value, np.int32):
            dtype = DataType.DT_INT32
        elif isinstance(value, np.int64):
            dtype = DataType.DT_INT64
        elif isinstance(value, int):
            dtype = DataType.DT_INT32
        elif isinstance(value, np.complex64):
            dtype = DataType.DT_COMPLEX64
        elif isinstance(value, np.complex128):
            dtype = DataType.DT_COMPLEX128
        elif isinstance(value, complex):
            dtype = DataType.DT_COMPLEX64
        elif isinstance(value, bool):
            dtype = DataType.DT_BOOL
        else:
            dtype = DataType.DT_INT32    

    dims = []
    for d in infered_shape:
        dims.append(TensorShapeProto.Dim(size=d))
    tensor_shape = TensorShapeProto(dim = dims)
    if dtype == DataType.DT_STRING:
        tensor_proto = TensorProto(dtype=dtype, tensor_shape=tensor_shape, string_val=values)
    else:
        np_dtype = _as_numpy_dtype(dtype)
        np_values = np.array(values, np_dtype)
        tensor_proto = TensorProto(dtype=dtype, tensor_shape=tensor_shape, tensor_content=np_values.tobytes())
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
    '''
    shape = [d.size for d in tensor_proto.tensor_shape.dim]
    num_elements = np.prod(shape, dtype=np.int64)
    np_dtype = _as_numpy_dtype(tensor_proto.dtype)
    
    if tensor_proto.tensor_content:
        return (np.frombuffer(tensor_proto.tensor_content,
                            dtype=np_dtype).copy().reshape(shape))

    if np_dtype == np.float16:
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
    elif np_dtype == np.str:
        values = list(tensor_proto.string_val)
        padding = num_elements - len(values)
        if padding > 0:
            last = values[-1] if values else ""
            values.extend([last] * padding)
        return np.array(values, dtype=np_dtype).reshape(shape)
    else:
        raise TypeError("Unsupported tensor type: %s" % tensor_proto.dtype)

    if values.size == 0:
        return np.zeros(shape, np_dtype)

    if values.size != num_elements :
        values = np.pad(values, (0, num_elements - values.size), "edge")

    return values.reshape(shape)
