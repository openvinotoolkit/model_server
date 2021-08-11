
<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GrpcClient`

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_model_metadata`

```python
get_model_metadata(request)
```

Send `GrpcModelMetadataRequest` to the server and return response.


**Args:**
 
 - <b>`request`</b>:  `GrpcModelMetadataRequest` object. 


**Returns:**
 `GrpcModelMetadataResponse` object 


**Raises:**
 
 - <b>`TypeError`</b>:   if provided argument is of wrong type.


**Examples:**
 
```python

config = {
    "address": "localhost",
    "port": 9000
}
client = make_grpc_client(config)
request = make_model_metadata_request("model")
response = client.get_model_metadata(request)

```

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_model_status`

```python
get_model_status(request)
```

Send `GrpcModelStatusRequest` to the server and return response. 


**Args:**
 
 - <b>`request`</b>:  `GrpcModelStatusRequest` object. 


**Returns:**
 `GrpcModelStatusResponse` object 


**Raises:**
 
 - <b>`TypeError`</b>:   if provided argument is of wrong type. 


**Examples:**

```python

config = {
    "address": "localhost",
    "port": 9000
}
client = make_grpc_client(config)
request = make_model_status_request("model")
response = client.get_model_status(request)

```


---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(request)
```

Send `GrpcPredictRequest` to the server and return response. 


**Args:**
 
 - <b>`request`</b>:  `GrpcPredictRequest` object. 


**Returns:**
 `GrpcPredictResponse` object 


**Raises:**
 
 - <b>`TypeError`</b>:   if provided argument is of wrong type. Many more for different serving reponses... 


**Examples:**

```python

config = {
"address": "localhost",
"port": 9000
}
client = make_grpc_client(config)
request = make_predict_request({"input": [1, 2, 3]}, "model")
response = client.predict(request)

```
