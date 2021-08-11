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
