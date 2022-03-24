<a href="../../lib/ovmsclient/tfs_compat/grpc/tensors.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_tensor_proto`

```python
make_tensor_proto(values, dtype=None, shape=None)
```

Creates `TensorProto` object from **values**. 


**Args:**
 
 - <b>`values`</b>:  Values to put in the `TensorProto`. 
 The accepted types are: python scalar, pythons list, numpy scalar and numpy ndarray. Python scalars and lists are internally converted to their numpy counterparts before further processing. Bytes values are placed in `TensorProto` <b>string_val</b> field.  


 - <b>`dtype`</b> <i>(optional)</i>:  tensor_pb2 DataType value.   If not provided, the function will try to infer the data type from **values** argument. 


 - <b>`shape`</b> <i>(optional)</i>:  The list or tuple of integers defining the shape of the tensor. If not provided, the function will try to infer the shape from **values** argument. 


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
with open("image.jpg", "rb") as f:
    data = f.read()
tensor_proto = make_tensor_proto(data)

```

---

<a href="README.md">Return to the main page</a>
