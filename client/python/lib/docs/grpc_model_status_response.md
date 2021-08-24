<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/responses.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GrpcModelStatusResponse`

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/responses.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict()
```


Return status in dictionary format:

```python

{
    ...
    <version_number>: {
        "state": <model_version_state>, 
        "error_code": <error_code>, 
        "error_message": <error_message>
    },          
    ...      
} 
```

---

<a href="README.md">Return to the main page</a>
