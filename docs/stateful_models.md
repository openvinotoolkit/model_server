# Serving Stateful Models {#ovms_docs_stateful_models}

## Stateless vs Stateful Models <a name="stateful_models"></a>

### Stateless model

A stateless model treats every inference request independently and does not recognize dependencies between consecutive inference requests. Therefore, it does not maintain a state between inference requests. Examples of stateless models could be image classification and object detection Convolutional Neural Networks (CNN).

### Stateful model

A stateful model recognizes dependencies between consecutive inference requests. It maintains a state between inference requests so that the next inference depends on the results of previous ones. Examples of stateful models could be online speech recognition models like Long Short Term Memory (LSTM).

---

**Note** that in the context of the Model Server, a model is considered stateful if it maintains state between **inference requests**. 

Some models might take the whole sequence of data as an input and iterate over the elements of that sequence internally, keeping the state between iterations. Such models are considered stateless since they perform inference on the whole sequence **in just one inference request**.

## Load and Serve Stateful Model <a name="stateful_serve"></a>

### Run Model Server with Stateful Model <a name="stateful_run"></a>

Serving stateful model in OpenVINO Model Server is very similar to serving stateless models. The only difference is that for stateful models you need to set `stateful` flag in the model configuration.

* Download and prepare example model from [rm_lstm4f](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/)

```bash
mkdir models && cd models
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/rm_lstm4f.counts
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/rm_lstm4f.nnet
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/rm_lstm4f.mapping
docker run -u $(id -u):$(id -g) -v $(pwd):/models:rw openvino/ubuntu20_dev:latest mo --framework kaldi --input_model /models/rm_lstm4f.nnet --counts /models/rm_lstm4f.counts --remove_output_softmax --output_dir /models/rm_lstm4f/1
```

* Starting OVMS with stateful model via command line:

```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/rm_lstm4f:/models/stateful_model -p 9000:9000 openvino/model_server:latest \
--port 9000 --model_path /models/stateful_model --model_name rm_lstm4f --stateful
```

* Starting OVMS with stateful model via config file:

```bash
echo '{
   "model_config_list":[
      {
         "config": {
            "name":"rm_lstm4f",
            "base_path":"/models/stateful_model",
            "stateful": true
         }
      }
   ]
}' >> config.json
```

```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/rm_lstm4f:/models/stateful_model -v $(pwd)/config.json:/models/config.json -p 9000:9000 openvino/model_server:latest \
--port 9000 --config_path /models/config.json
```

 Optionally, you can also set additional parameters specific for stateful models. 
 
 ### Configuration Options for Stateful Models <a name="stateful_params"></a>

**Model configuration**:

| Option  | Value format  | Description  | Default value |
|---|---|---|---|
| `stateful` | `bool` | If set to true, model is loaded as stateful. | false |
| `idle_sequence_cleanup` | `bool` | If set to true, model will be subject to periodic sequence cleaner scans. <br> See [idle sequence cleanup](#stateful_cleanup). | true |
| `max_sequence_number` | `uint32` | Determines how many sequences can be  handled concurrently by a model instance. | 500 |
| `low_latency_transformation` | `bool` | If set to true, model server will apply [low latency transformation](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_model_state_intro.html#lowlatency-transformations) on model load. | false |

**Note:** Setting `idle_sequence_cleanup`, `max_sequence_number` and `low_latency_transformation` require setting `stateful` to true.

**Server configuration**:

| Option  | Value format  | Description  | Default value |
|---|---|---|---|
| `sequence_cleaner_poll_wait_minutes` | `uint32` | Time interval (in minutes) between next sequence cleaner scans. Sequences of the models that are subjects to idle sequence cleanup that have been inactive since the last scan are removed. Zero value disables sequence cleaner.<br> See [idle sequence cleanup](#stateful_cleanup). | 5 |

See also [all server and model configuration options](parameters.md) to have a complete setup.

## Run Inference on Stateful Model <a name="stateful_inference"></a>

### Special Inputs for Sequence Handling <a name="stateful_inputs"></a>

Stateful model works on consecutive inference requests that are associated with each other and form a **sequence** of requests. A single stateful model can handle multiple independent sequences at a time. When the model server receives requests for the stateful model, it maps each request to the proper sequence and its memory state. OVMS also tracks the beginning and the end of the sequence to properly manage system resources.

Requests to stateful models must contain additional inputs besides the data for prediction:
- `sequence_id` - which is a 64-bit unsigned integer identifying the sequence (unique in the scope of the model instance). Value 0 is equivalent to not providing this input at all.
- `sequence_control_input` - which is 32-bit unsigned integer indicating sequence start and end. Accepted values are: 
   - 0 - no control input (has no effect - equivalent to not providing this input at all)
   - 1 - indicates the beginning of the sequence
   - 2 - indicates the end of the sequence

**Note**: Model server also appends `sequence_id` to every response - the name and format of `sequence_id` output is the same as in `sequence_id` input.

**Both `sequence_id` and `sequence_control_input` shall be provided as tensors with 1 element array (shape:[1]) and appropriate precision.**  
_See examples for gRPC and HTTP below_.

In order to successfully infer the sequence, perform these actions:
1. **Send the first request in the sequence and signal sequence start.**

   To start the sequence you need to add `sequence_control_input` with the value of 1 to your request's inputs. You can also:
      - add `sequence_id` with the value of your choice or
      - add `sequence_id` with 0 or do not add `sequence_id` at all - in this case, the Model Server will provide a unique id for the sequence and since it will be appended to the outputs, you will be able to read it and use with the next requests. 

      If the provided `sequence_id` is already occupied, OVMS will return an [error](#error-codes) to avoid conflicts.

2. **Send remaining requests except the last one.**

   To send requests in the middle of the sequence you need to add `sequence_id` of your sequence. In this case, `sequence_id` is mandatory and not providing this input or setting its value to 0 is not allowed.

   In this case `sequence_control_input` must be empty or 0.

3. **Send the last request in the sequence and signal sequence end.**

   To end the sequence you need to add `sequence_control_input` with the value of 2 to your request's inputs. You also need to add `sequence_id` of your sequence. In this case, `sequence_id` is mandatory and not providing this input or setting its value to 0 is not allowed.


### Inference via gRPC <a name="stateful_grpc"></a>

Inference on stateful models via gRPC is very similar to inference on stateless models (_see [gRPC API](model_server_grpc_api_tfs.md) for reference_). The difference is that requests to stateful models must contain additional inputs with information necessary for proper sequence handling.

`sequence_id` and `sequence_control_input` must be added to gRPC request inputs as [TensorProtos](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto).

* For `sequence_id` model server expects one value in tensor proto [uint64_val](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto#L85) field.

* For `sequence_control_input` model server expects one value in tensor proto [uint32_val](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto#L82) field.

Both inputs must have `TensorShape` set to [1] and appropriate `DataType`:
- `DT_UINT64` for `sequence_id`
- `DT_UINT32` for `sequence_control_input`

Example: (_using Python tensorflow and tensorflow-serving-api packages_):

```
...
import grpc

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import make_tensor_proto, make_ndarray, expand_dims
from tensorflow_serving.apis import predict_pb2

...

SEQUENCE_START = 1
SEQUENCE_END = 2
sequence_id = 10

channel = grpc.insecure_channel("localhost:9000")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "stateful_model"

""" 
Add inputs with data to infer
"""

################   Add stateful specific inputs   #################

################ Starting sequence with custom ID #################

request.inputs['sequence_control_input'].CopyFrom(
               make_tensor_proto([SEQUENCE_START], dtype="uint32"))
request.inputs['sequence_id'].CopyFrom(
                make_tensor_proto([sequence_id], dtype="uint64"))


################   Starting sequence without ID   #################

request.inputs['sequence_control_input'].CopyFrom(
               make_tensor_proto([SEQUENCE_START], dtype="uint32"))


################       Non control requests       #################

request.inputs['sequence_id'].CopyFrom(
               make_tensor_proto([sequence_id], dtype="uint64"))


################         Ending sequence          #################

request.inputs['sequence_control_input'].CopyFrom(
               make_tensor_proto([SEQUENCE_END], dtype="uint32"))
request.inputs['sequence_id'].CopyFrom(
                make_tensor_proto([sequence_id], dtype="uint64"))

###################################################################

# Send request to OVMS and get response
response = stub.Predict(request, 10.0)

# response variable now contains model outputs (inference results) as well as sequence_id in response.outputs

# Fetch sequence id from the response
sequence_id = response.outputs['sequence_id'].uint64_val[0]

```

See [grpc_stateful_client.py](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/speech_recognition_with_kaldi_model/python/grpc_stateful_client.py) example client for reference.

### Inference via HTTP <a name="stateful_http"></a>

Inference on stateful models via HTTP is very similar to inference on stateless models (_see [REST API](model_server_rest_api_tfs.md) for reference_). The difference is that requests to stateful models must contain additional inputs with information necessary for proper sequence handling.

`sequence_id` and `sequence_control_input` must be added to HTTP request by adding new `key:value` pair in `inputs` field of JSON body. 

For both inputs, the value must be a single number in a 1-dimensional array.

Example: (_using Python requests package_):

```
...
import json
import requests
...

SEQUENCE_START = 1
SEQUENCE_END = 2
sequence_id = 10


inputs = {}

""" 
Add inputs with data to infer
"""

################   Add stateful specific inputs   #################

################ Starting sequence with custom ID #################

inputs['sequence_control_input'] = [int(SEQUENCE_START)]
inputs['sequence_id'] = [int(sequence_id)]


################   Starting sequence without ID   #################

inputs['sequence_control_input'] = [int(SEQUENCE_START)]


################       Non control requests       #################

inputs['sequence_id'] = [int(sequence_id)]


################         Ending sequence          #################

inputs['sequence_control_input'] = [int(SEQUENCE_END)]
inputs['sequence_id'] = [int(sequence_id)]

###################################################################

# Prepare request
signature = "serving_default"
request_body = json.dumps({"signature_name": signature,'inputs': inputs})

# Send request to OVMS and get response
response = requests.post("localhost:5555/v1/models/stateful_model:predict", data=request_body)

# Parse response
response_body = json.loads(response.text)

# response_body variable now contains model outputs (inference results) as well as sequence_id in response_body["outputs"]

# Fetch sequence id from the response
sequence_id = response_body["outputs"]["sequence_id"]

```
See [rest_stateful_client.py](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/speech_recognition_with_kaldi_model/python/rest_stateful_client.py) example client for reference.

### Error Codes <a name="stateful_errors"></a>

When a request is invalid or could not be processed, you can expect following errors specific to inference on stateful models:

| Description  | gRPC | HTTP |
|---|---|---|
| Sequence with a provided ID does not exist. | NOT_FOUND | 404 NOT FOUND |
| Sequence with a provided ID already exists.  | ALREADY_EXISTS | 409 CONFLICT |
| Server received SEQUENCE START request with ID of the sequence that is set for termination, but the last request of that sequence is still being processed. | FAILED_PRECONDITION | 412 PRECONDITION FAILED |
| Max sequence number has been reached. Could not create a new sequence. | UNAVAILABLE | 503 SERVICE UNAVAILABLE | 
| Sequence ID has not been provided in request inputs. | INVALID_ARGUMENT | 400 BAD REQUEST |
| Unexpected value of sequence control input. | INVALID_ARGUMENT | 400 BAD REQUEST |
| Could not find sequence id in expected tensor proto field uint64_val. | INVALID_ARGUMENT | N/A |
| Could not find sequence control input in expected tensor proto field uint32_val. | INVALID_ARGUMENT | N/A |
| Special input proto does not contain tensor shape information. | INVALID_ARGUMENT | N/A |

## Idle Sequence Cleanup <a name="stateful_cleanup"></a>

Once started sequence might get dropped for some reason like lost connection etc. In this case model server will not receive SEQUENCE_END signal and will not free sequence resources. To prevent keeping idle sequences indefinitely, the Model Server launches a sequence cleaner thread that periodically scans stateful models and checks if their sequences received any valid inference request recently. If not, such sequences are removed, their resources are freed and their ids can be reused.

Two parameters regulate sequence cleanup. 
One is `sequence_cleaner_poll_wait_minutes` which holds the value of the time interval between the next scans. If there has been not a single valid request with a particular sequence id between two consecutive checks, the sequence is considered idle and gets deleted. 

`sequence_cleaner_poll_wait_minutes` is a server parameter and is common for all models. By default, the time between two consecutive cleaner scans is set to 5 minutes. Setting this value to 0 disables sequence cleaner.


Stateful models can either be subject to idle sequence cleanup or not.
You can set this **per model** with `idle_sequence_cleanup` parameter. 
If set to `true` sequence cleaner will check that model. Otherwise, sequence cleaner will skip that model, and its inactive sequences will not get removed. By default, this value is set to `true`.

## Known Limitations <a name="stateful_limitations"></a>

There are limitations for using stateful models with OVMS:

 - Support inference execution only using CPU as the target device.
 - Support Kaldi models with memory layers and non-Kaldi models with Tensor Iterator. See this [docs about stateful networks](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_model_state_intro.html) to learn about stateful networks representation in OpenVINO.
 - [Auto batch size and shape](shape_batch_size_and_layout.md) are **not** available in stateful models.
 - Stateful model instances **cannot** be used in [DAGs](dag_scheduler.md).
 - Requests ordering is guaranteed only when a single client sends subsequent requests in a synchronous manner. Concurrent interaction with the same sequence might negatively affect the accuracy of the results.
 - When stateful model instance gets reloaded due to change in model configuration, **all ongoing sequences are dropped**.
 - Model type cannot be changed in the runtime - switching stateful flag will be rejected.
