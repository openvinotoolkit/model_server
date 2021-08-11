<!-- markdownlint-disable -->

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/tensors.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tfs_compat.grpc.tensors`


<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/tensors.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_tensor_proto`

```python
make_tensor_proto(values, dtype=None, shape=None)
```

Creates `TensorProto` object from **values**. 


**Args:**
 
 - <b>`values`</b>:  Values to put in the `TensorProto`. 
 The accepted types are: python scalar, pythons list, numpy scalar and numpy ndarray. Python scalars and lists are internally converted to their numpy counterparts before further processing. Bytes values are placed in `TensorProto` <b>string_val</b> field.  


 - <b>`dtype`</b> (optional):  tensor_pb2 DataType value.   If not provided, the function will try to infer the data type from **values** argument. 


 - <b>`shape`</b> (optional):  The list or tuple of integers defining the shape of the tensor. If not provided, the function will try to infer the shape from **values** argument. 


**Returns:**
 TensorProto object filled with **values**. 


**Raises:**
 
 - <b>`TypeError`</b>:   if unsupported types are provided. 
 - <b>`ValueError`</b>:  if arguments have inappropriate values. 


**Examples:**

```python

# With python list
data = [[1, 2, 3], [4, 5, 6]]
tensor_proto = make_tensor_proto(data)


# With numpy array:
data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
tensor_proto = make_tensor_proto(data)


# With binary data:
data = bytes([1, 2, 3, 4, 5, 6])
tensor_proto = make_tensor_proto(data)

```

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/tensors.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_ndarray`

```python
make_ndarray(tensor_proto)
```

Create `numpy ndarray` from **tensor_proto**. 


**Args:**
 
 - <b>`tensor_proto`</b>:  `TensorProto` object. 


**Returns:**
 `Numpy ndarray` with tensor proto contents. 


**Raises:**
 
 - <b>`TypeError`</b>:   if unsupported type is provided. 



**Examples:**

 Create `TensorProto` with `make_tensor_proto` and convert it back to numpy array with `make_ndarray`: 

```python

data = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
tensor_proto = make_tensor_proto(data)
output = make_ndarray(tensor_proto)
    
```
