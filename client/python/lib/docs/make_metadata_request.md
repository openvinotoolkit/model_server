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
