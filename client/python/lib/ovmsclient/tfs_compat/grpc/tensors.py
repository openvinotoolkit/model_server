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

from tensorflow.core.framework.tensor_pb2 import TensorProto

def make_tensor_proto(values, dtype=None, shape=None):
    '''
    Create TensorProto object from values.

    Args:

        values: Values to put in the TensorProto.
            The accepted types are: python scalar, pythons list, numpy scalar and numpy ndarray.
            Python scalars and lists are internally converted to their numpy counterparts before further processing.
            String values are converted to bytes and placed in TensorProto string_val field. 

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

    raise NotImplementedError

def as_numpy_dtype(tensor_dtype):
    if tensor_dtype == 0:
        raise ValueError("Tensor data type invalid")
    elif tensor_dtype == 1:
        return np.float32
    elif tensor_dtype == 2:
        return np.float64
    elif tensor_dtype == 3:
        return np.int32
    elif tensor_dtype == 4:
        return np.uint8
    elif tensor_dtype == 5:
        return np.int16
    elif tensor_dtype == 6:
        return np.int8
    elif tensor_dtype == 7:
        return np.str
    elif tensor_dtype == 8:
        return np.complex64
    elif tensor_dtype == 9:
        return np.int64
    elif tensor_dtype == 10:
        return np.bool
    elif tensor_dtype == 17:
        return np.uint16
    elif tensor_dtype == 22:
        return np.uint32
    elif tensor_dtype == 23:
        return np.uint64
    else:
        return np.uint32


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
    tensor_shape = [d.size for d in tensor_proto.tensor_shape.dim]
    size = np.prod(tensor_shape, dtype=np.int64)
    dtype = as_numpy_dtype(tensor_proto.dtype)
    
    if tensor_proto.tensor_content:
        return (np.frombuffer(tensor_proto.tensor_content,
                            dtype=dtype).copy().reshape(tensor_shape))

    if dtype == np.str:
        values = list(tensor_proto.string_val)
        padding = size - len(values)
        if padding > 0:
            last = values[-1] if values else ""
            values.extend([last] * padding)
        return np.array(values, dtype=dtype).reshape(tensor_shape)

    if dtype == np.float16:
        values = np.fromiter(tensor_proto.half_val, dtype=np.uint16)
        values.dtype = dtype
    elif dtype == np.float32:
        values = np.fromiter(tensor_proto.float_val, dtype=dtype)
    elif dtype == np.float64:
        values = np.fromiter(tensor_proto.double_val, dtype=dtype)
    elif dtype in [
        np.int32, np.uint8, np.uint16, np.int16, np.int8,
        np.qint32, np.quint8, np.qint8, np.qint16, np.quint16
    ]:
        values = np.fromiter(tensor_proto.int_val, dtype=dtype)
    elif dtype == np.int64:
        values = np.fromiter(tensor_proto.int64_val, dtype=dtype)
    elif dtype == np.uint32:
        values = np.fromiter(tensor_proto.uint32_val, dtype=dtype)
    elif dtype == np.uint64:
        values = np.fromiter(tensor_proto.uint64_val, dtype=dtype)
    elif dtype == np.complex64:
        it = iter(tensor_proto.scomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
    elif dtype == np.complex128:
        it = iter(tensor_proto.dcomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
    elif dtype == np.bool:
        values = np.fromiter(tensor_proto.bool_val, dtype=dtype)
    else:
        raise TypeError("Unsupported tensor type: %s" % tensor_proto.dtype)

    if values.size == 0:
        return np.zeros(tensor_shape, dtype)

    if values.size != size:
        values = np.pad(values, (0, size - values.size), "edge")

    return values.reshape(tensor_shape)
