# Serving stateful models with OpenVINO Model Server

## Table of contents

* [Stateless vs stateful models](#stateful_models)
* [Load and serve stateful model](#stateful_serve)
* [Run inference on stateful model](#stateful_inference)
    * [Special inputs for sequence handling](#stateful_inputs)
    * [Inference via gRPC](#stateful_grpc)
    * [Inference via HTTP](#stateful_rest)
* [Known limitations](#stateful_limitations)

## Stateless vs stateful models <a name="stateful_models"></a>

### Stateless model

A stateless model treats every inference request independently and does not recognize dependencies between consecutive inference requests. Therefore it  does not maintain state between inference requests. Examples of stateless models could be image classification and object detection CNNs.

### Stateful model

A stateful model recognizes dependencies between consecutive inference requests. It maintains state between inference requests so that next inference depends on the results of previous ones. Examples of stateful models could be online speech recogniton models like LSTMs.

Note that in the context of model server, model is considered stateful if it maintains state between **inference requests**. 

Some models might take the whole sequence of data as an input and iterate over the elements of that sequence internally, keeping the state between iterations. Such models are considered stateless since they perform inference on the whole sequence **in just one inference request**.


## Load and serve stateful model <a name="stateful_serve"></a>

Serving stateful model in OpenVINO Model Server is very similar to serving stateless models. The only difference is that for stateful models you need to set `stateful` flag in model configuration.

* Starting OVMS with stateful model via command line:

```
docker run -d -u $(id -u):$(id -g) -v <host_model_path>:/models/stateful_model -p 9000:9000 openvino/model_server:latest \ 
--port 9000 --model_path /models/stateful_model --model_name stateful_model --stateful
```

* Starting OVMS with stateful model via config file:

```
{
   "model_config_list":[
      {
         "config": {
            "name":"stateful_model",
            "base_path":"/models/stateful_model",
            "stateful": true
         }
      }
   ]
}
```

```
docker run -d -u $(id -u):$(id -g) -v <host_model_path>:/models/stateful_model -v <host_config_path>:/models/config.json -p 9000:9000 openvino/model_server:latest \ 
--port 9000 --config_path /models/config.json
```

 Optionally, you can also set additional parameters specific for stateful models described below:

| Option  | Value format  | Description  | Default value |
|---|---|---|---|
| `stateful` | `bool` | If set to true, model is loaded as stateful | false |
| `sequence_timeout_seconds` | `uint32` | Determines how long sequence can be idle (in seconds) before model server removes it | 60 |
| `max_sequence_number` | `uint32` | Determines how many sequences can be handled concurrently by a model instance | 500 |
| `low_latency_transformation` | `bool` | If set to true, model server will apply low latency transformation on model load (see https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_network_state_intro.html#lowlatency_transformation for reference) | false |

**Note:** Setting `sequence_timeout_seconds`, `max_sequence_number` and `low_latency_transformation` require setting `stateful` to true.


## Run inference on stateful model <a name="stateful_inference"></a>
