import numpy as np
from pyovms import Tensor
import pytest

NUMPY_NUMERICAL_TYPES = [np.bool_, np.uint8, np.uint16, np.uint32, np.uint64,
                        np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]

@pytest.mark.parametrize("array_dtype", NUMPY_NUMERICAL_TYPES)
def test_creating_from_numpy_numerical_array(array_dtype):
    npy_arr = np.array(np.ones((1,3,300,300), dtype=array_dtype))
    ovms_py_tensor = Tensor(npy_arr)
    assert npy_arr.shape == ovms_py_tensor.shape
    assert npy_arr.data.format == ovms_py_tensor.data.format
    assert npy_arr.itemsize == ovms_py_tensor.itemsize
    assert npy_arr.strides == ovms_py_tensor.strides
    assert npy_arr.data == ovms_py_tensor.data

@pytest.mark.parametrize("data", ["single_string", ["batch", "of", "strings"]])
def test_creating_from_numpy_string_array(data):
    npy_arr = np.array(data)
    ovms_py_tensor = Tensor(npy_arr)
    assert npy_arr.shape == ovms_py_tensor.shape
    assert npy_arr.data.format == ovms_py_tensor.data.format
    assert npy_arr.itemsize == ovms_py_tensor.itemsize
    assert npy_arr.strides == ovms_py_tensor.strides
    assert npy_arr.data.tobytes() == ovms_py_tensor.data.tobytes()

def test_creating_from_python_bytes():
    data = "some string encoded with default UTF-8 encoding".encode()
    ovms_py_tensor = Tensor(data)
    assert ovms_py_tensor.ndim == 1
    assert ovms_py_tensor.shape == (len(data),)
    assert ovms_py_tensor.data.format == "B" # UINT8
    assert ovms_py_tensor.itemsize == 1
    assert ovms_py_tensor.strides == (1,)
    assert ovms_py_tensor.data.tobytes() == data
