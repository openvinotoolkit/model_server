<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/responses.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GrpcPredictResponse`

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/responses.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict()
```
Returns inference results in dictionary format: 

``` python

{
    ...          
    <output_name>: <numpy ndarray with result>      
    ...      
} 

```

---

<a href="README.md">Return to the main page</a>
