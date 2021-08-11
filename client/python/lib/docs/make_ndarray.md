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
