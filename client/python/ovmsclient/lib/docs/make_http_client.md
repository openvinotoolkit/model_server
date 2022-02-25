<a href="../../lib/ovmsclient/tfs_compat/http/serving_client.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_http_client`

```python
make_http_client(url, tls_config)
```

**Description**:

Creates [`HttpClient`](http_client.md) object. 


**Args:**
 
 - <b>`url`</b> - Model Server URL as a string in format `<address>:<port>`
 - <b>`tls_config`</b> <i>(optional)</i>: dictionary with TLS configuration. The accepted format is: 

    ```python

    {                   
        "client_key_path": <Path to client key file>,
        "client_cert_path": <Path to client certificate file>,
        "server_cert_path": <Path to server certificate file>             
     }                       

    ```

    With following types accepted: 
    | Key | Value type |
    |---|---|
    | client_key_path | string |
    | client_cert_path | string |
    | server_cert_path | string |
                                    
    By default TLS is not used and `tls_config` value is `None`.


**Returns:**
 [`HttpClient`](http_client.md) object 



**Raises:**
 
 - <b>`ValueError, TypeError`</b>:   if provided config is invalid. 



**Examples:**

 Create minimal `HttpClient`: 

```python

from ovmsclient import make_http_client
client = make_http_client("localhost:9000")

```

Create `HttpClient` with TLS:

```python

from ovmsclient import make_http_client

tls_config = {
    "tls_config": {
        "client_key_path": "/opt/tls/client.key",
        "client_cert_path": "/opt/tls/client.crt",
        "server_cert_path": "/opt/tls/server.crt"    
    }
}

client = make_http_client("localhost:9000", tls_config=tls_config)

```

---

<a href="README.md">Return to the main page</a>
