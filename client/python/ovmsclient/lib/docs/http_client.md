
<a href="../../lib/ovmsclient/tfs_compat/http/serving_client.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HttpClient`

---

<a href="../../lib/ovmsclient/tfs_compat/http/serving_client.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_model_metadata`

```python
get_model_metadata(model_name, model_version, timeout)
```

Request model metadata.


**Args:**
 
 - <b>`model_name`</b>:  name of the requested model. Accepted types: `string`.
 - <b>`model_version`</b> <i>(optional)</i>: version of the requested model. Accepted types: `positive integer`. Value 0 is special and means the latest served version will be chosen <i>(only in OVMS, TFS requires specific version number provided)</i>. Default value: 0.
 - <b>`timeout`</b> <i>(optional)</i>: time in seconds to wait for the response from the server. If exceeded, TimeoutError is raised. 
 Accepted types: `positive integer`, `positive float`. Value 0 is not accepted. Default value: 10.0.


**Returns:**
 Dictionary with model metadata in form:

 ``` python

{
    "model_version": <version_number>,
    "inputs": {
        <input_name>: {
            "shape": <input_shape>,
            "dtype": <input_dtype>,
        },
        ...
    },
    "outputs":
        <output_name>: {
            "shape": <output_shape>,
            "dtype": <output_dtype>,
        },
        ...
    }
} 

``` 


**Raises:**
 
- <b>`TypeError`</b>:  if provided argument is of wrong type.
- <b>`ValueError`</b>: if provided argument has unsupported value.
- <b>`ConnectionError`</b>: if there is an issue with server connection.
- <b>`TimeoutError`</b>: if request handling duration exceeded timeout.
- <b>`ModelNotFound`</b>: if model with specified name and version does not exist
                           in the model server.
- <b>`BadResponseError`</b>: if server response in malformed and cannot be parsed.


**Examples:**
 
```python
import ovmsclient
client = ovmsclient.make_http_client("localhost:9000")
# request metadata of the specific model version, with timeout set to 2.5 seconds
model_metadata = client.get_model_metadata(model_name="model", model_version=1, timeout=2.5)
# request metadata of the latest model version
model_metadata = client.get_model_metadata(model_name="model")

```

---

<a href="../../lib/ovmsclient/tfs_compat/http/serving_client.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_model_status`

```python
get_model_status(model_name, model_version, timeout)
```

Request model status.


**Args:**
 
 - <b>`model_name`</b>:  name of the requested model. Accepted types: `string`.
 - <b>`model_version`</b> <i>(optional)</i>: version of the requested model. Accepted types: `positive integer`. Value 0 means that status of all model versions will be returned. Default value: 0.
 - <b>`timeout`</b> <i>(optional)</i>: time in seconds to wait for the response from the server. If exceeded, TimeoutError is raised. 
 Accepted types: `positive integer`, `positive float`. Value 0 is not accepted. Default value: 10.0.


**Returns:**
 Dictionary with model status in form:

 ``` python

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


**Raises:**
 
- <b>`TypeError`</b>:  if provided argument is of wrong type.
- <b>`ValueError`</b>: if provided argument has unsupported value.
- <b>`ConnectionError`</b>: if there is an issue with server connection.
- <b>`TimeoutError`</b>: if request handling duration exceeded timeout.
- <b>`ModelNotFound`</b>: if model with specified name and version does not exist
                          in the model server.
- <b>`BadResponseError`</b>: if server response in malformed and cannot be parsed.


**Examples:**
 
```python
import ovmsclient
client = ovmsclient.make_http_client("localhost:9000")
# request status of the specific model version, with timeout set to 2.5 seconds
model_status = client.get_model_status(model_name="model", model_version=1, timeout=2.5)
# request status of all model versions
model_status = client.get_model_status(model_name="model")

```

---

<a href="../../lib/ovmsclient/tfs_compat/http/serving_client.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(inputs, model_name, model_version, timeout)
```

Request prediction on provided inputs. 


**Args:**
 
- <b>`inputs`</b>: dictionary in form
    ```python
    {
        ...
        <input_name>:<input_data>
        ...
    }
    ```               
    Following types are accepted: 

    | Key | Value type |
    |---|---|
    | input_name | string |
    | input_data | python scalar, python list, numpy scalar, numpy array |

 - <b>`model_name`</b>:  name of the requested model. Accepted types: `string`.
 - <b>`model_version`</b> <i>(optional)</i>: version of the requested model. Accepted types: `positive integer`. Value 0 is special and means the latest served version will be chosen <i>(only in OVMS, TFS requires specific version number provided)</i>. Default value: 0.
 - <b>`timeout`</b> <i>(optional)</i>: time in seconds to wait for the response from the server. If exceeded, TimeoutError is raised. 
 Accepted types: `positive integer`, `positive float`. Value 0 is not accepted. Default value: 10.0.


**Returns:**
 - if model has one output: `numpy ndarray` with prediction results
 - if model has multiple outputs: `dictionary` in form:
     ```python
    {
        ...
        <output_name>:<prediction_result>
        ...
    }
    ```   
    Where `output_name` is a `string` and `prediction_result` is a `numpy ndarray`. Both strings and binary data are returned as array of `numpy.bytes_` dtype. 


**Raises:**

- <b>`TypeError`</b>:  if provided argument is of wrong type.
- <b>`ValueError`</b>: if provided argument has unsupported value.
- <b>`ConnectionError`</b>: if there is an issue with server connection.
- <b>`TimeoutError`</b>: if request handling duration exceeded timeout.
- <b>`ModelNotFound`</b>: if model with specified name and version does not exist
                          in the model server.
- <b>`BadResponseError`</b>: if server response in malformed and cannot be parsed.
- <b>`InvalidInputError`</b>: if provided inputs do not match model's inputs


**Examples:**

```python
import ovmsclient
client = ovmsclient.make_http_client("localhost:9000")

# Numeric input
inputs = {"input": [1, 2, 3]}
# request prediction on specific model version, with timeout set to 2.5 seconds
results = client.predict(inputs=inputs, model_name="model", model_version=1, timeout=2.5)
# request prediction on the latest model version
results = client.predict(inputs=inputs, model_name="model")

# String input
inputs = {"input": ["We have a really nice", "One, two, three,"]}
results = client.predict(inputs=inputs, model_name="model")
print(inputs["input"])
# ['We have a really nice', 'One, two, three,']
print(results)
# [b'We have a really nice way' b'One, two, three, four']
```

---

<a href="README.md">Return to the main page</a>
