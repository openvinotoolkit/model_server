<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/responses.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GrpcModelMetadataResponse`

---

<a href="../../../../client/python/lib/ovmsclient/tfs_compat/grpc/responses.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict()
```
Returns metadata in dictionary format: 

``` python

{
    ...          
    <version_number>: 
        {
            "inputs": {
                <input_name>: {
                    "shape": <input_shape>,
                    "dtype": <input_dtype>,
                    },                      
                ...              
            },           
            "outputs": {
                <output_name>: {
                    "shape": <output_shape>,
                    "dtype": <output_dtype>,
                    },
                ...              
            }
        },          
        ...      
} 

```

---

<a href="README.md">Return to the main page</a>
