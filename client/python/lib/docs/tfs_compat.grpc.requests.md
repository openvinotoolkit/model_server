<!-- markdownlint-disable -->

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/requests.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tfs_compat.grpc.requests`

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/requests.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_predict_request`

```python
make_predict_request(inputs, model_name, model_version=0)
```

Create `GrpcPredictRequest` object. 


**Args:**
 
 - <b>`inputs`</b>:  Python dictionary in format 
    ```python
    {<input_name>:<input_data>}
    ```               
    Following types are accepted: 

    | Key | Value type |
    |---|---|
    | input_name | string |
    | input_data | python scalar, python liste, numpy scalar, numpy array, TensorProto |        

    If provided **input_data** is not TensorProto, the `make_tensor_proto` function with default parameters will be called internally. 

- <b>`model_name`</b>: Name of the model that will receive the request. 

- <b>`model_version`</b> <i>(optional)</i>: Version of the model that will receive the request.          By default this value is set to 0, meaning the request will be sent to the default version of the model. 


**Returns:**
 `GrpcPredictRequest` object filled with **inputs** and target model spec. 


**Raises:**
 
 - <b>`TypeError`</b>:   if unsupported types are provided. 
 - <b>`ValueError`</b>:  if arguments have inappropriate values. 


**Examples:**

 Request to the default version of the model called "model" that has 2 inputs:   

```python 
predict_request = make_predict_request(
    inputs={
        "binary_input": bytes([1, 2, 3, 4, 5, 6]),
        "numeric_input": np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    }, 
    model_name="model"
)
```

Request to the second version of the model called "model" that has 1 input.
Providing data as `TensorProto` to make sure desired data type is set for the input:

```python
predict_request = make_predict_request(
    inputs={
        "input": make_tensor_proto([1, 2, 3], dtype=DataTypes.float32)
    }, 
    model_name="model", 
    model_version=2
)
```


---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/requests.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_metadata_request`

```python
make_metadata_request(model_name, model_version=0)
```

Create `GrpcModelMetadataRequest` object. 


**Args:**
 
 - <b>`model_name`</b>:  Name of the model that will receive the request. 

 - <b>`model_version`</b> (optional):  Version of the model that will receive the request.  By default this value is set to 0, meaning the request will be sent to the default version of the model. 


**Returns:**
 `GrpcModelMetadataRequest` object with target model spec. 


**Raises:**
 
 - <b>`TypeError`</b>:   if unsupported types are provided. 
 - <b>`ValueError`</b>:  if arguments have inappropriate values. 


**Examples:**

 Request to the second version of the model called "model":   

```python
metadata_request = make_metadata_request(model_name="model", model_version=2)
```

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/requests.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_status_request`

```python
make_status_request(model_name, model_version=0)
```

Create `GrpcModelStatusRequest` object. 



**Args:**
 


 - <b>`model_name`</b>:  Name of the model that will receive the request. 


 - <b>`model_version`</b> (optional):  Version of the model that will receive the request. Must be type int.  By default this value is set to 0, meaning the request will be sent to the default version of the model. 



**Returns:**
 `GrpcModelStatusRequest` object with target model spec. 



**Raises:**
 
 - <b>`TypeError`</b>:   if unsupported types are provided. 
 - <b>`ValueError`</b>:  if arguments have inappropriate values. 



**Examples:**

 Request to the second version of the model called "model":   

```python 
status_request = make_status_request(model_name="model", model_version=2)
```
