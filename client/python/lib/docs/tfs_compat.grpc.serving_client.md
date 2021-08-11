<!-- markdownlint-disable -->

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tfs_compat.grpc.serving_client`



---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/serving_client.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_grpc_client`

```python
make_grpc_client(config)
```

Creates `GrpcClient` object. 



**Args:**
 
 - <b>`config`</b>:  Python dictionary with client configuration. The accepted format is: 

    ```python

    {
        "address": <IP address of the serving>,                  
        "port": <Port number used by the gRPC interface of the server>,                      
        "tls_config": {                      
            "client_key_path": <Path to client key file>,
            "client_cert_path": <Path to client certificate file>,
            "server_cert_path": <Path to server certificate file>
        }              
    }                        

    ```

    With following types accepted: 
    | Key | Value type |
    |---|---|
    | address | string |
    | port | integer |
    | client_key_path | string |
    | client_cert_path | string |
    | server_cert_path | string |
                                    
    The minimal config must contain `address` and `port`. 


**Returns:**
 `GrpcClient` object 



**Raises:**
 
 - <b>`ValueError, TypeError`</b>:   if provided config is invalid. 



**Examples:**

 Create minimal `GrpcClient`: 

```python

config = {
    "address": "localhost",
    "port": 9000
}
client = make_grpc_client(config)

```

Create GrpcClient with TLS:

```python

config = {
    "address": "localhost",
    "port": 9000,
    "tls_config": {
        "client_key_path": "/opt/tls/client.key",
        "client_cert_path": "/opt/tls/client.crt",
        "server_cert_path": "/opt/tls/server.crt"    
    }
}
client = make_grpc_client(config)

```


---

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
