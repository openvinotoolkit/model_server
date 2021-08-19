
<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GrpcClient`

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_model_metadata`

```python
get_model_metadata(request)
```

Send [`GrpcModelMetadataRequest`](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/lib/ovmsclient/tfs_compat/grpc/requests.py#L31) to the server and return response.


**Args:**
 
 - <b>`request`</b>:  `GrpcModelMetadataRequest` object. 


**Returns:**
 `GrpcModelMetadataResponse` object 


**Raises:**
 
 - <b>`TypeError`</b>:   if request argument is of wrong type.
 - <b>`ValueError`</b>:   if request argument has invalid contents.
 - <b>`ConnectionError`</b>:   if there was an error while sending request to the server.


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

Send [`GrpcModelStatusRequest`](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/lib/ovmsclient/tfs_compat/grpc/requests.py#L37) to the server and return response. 


**Args:**
 
 - <b>`request`</b>:  `GrpcModelStatusRequest` object. 


**Returns:**
 `GrpcModelStatusResponse` object 


**Raises:**
 
 - <b>`TypeError`</b>:   if request argument is of wrong type.
 - <b>`ValueError`</b>:   if request argument has invalid contents.
 - <b>`ConnectionError`</b>:   if there was an error while sending request to the server.


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

Send [`GrpcPredictRequest`](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/lib/ovmsclient/tfs_compat/grpc/requests.py#L25) to the server and return response. 


**Args:**
 
 - <b>`request`</b>:  `GrpcPredictRequest` object. 


**Returns:**
 `GrpcPredictResponse` object 


**Raises:**
 
 - <b>`TypeError`</b>:   if request argument is of wrong type.
 - <b>`ValueError`</b>:   if request argument has invalid contents.
 - <b>`ConnectionError`</b>:   if there was an error while sending request to the server.


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

---

<a href="README.md">Return to the main page</a>
