# Metrics {#ovms_docs_metrics}

## Introduction

This document describes how to use metrics endpoint in the OpenVINO Model Server. They can be applied for:

- Providing performance and utilization statistics for monitoring and benchmarking purposes

- Auto scaling of the model server instances in Kubernetes and OpenShift based on application related metrics

> **NOTE**: currently, Metrics feature is released as a preview feature.

Built-in metrics allow tracking the performance without any extra logic on the client side or using network traffic monitoring tools like load balancers or reverse-proxies.

It also exposes metrics which are not related to the netowrk traffic. 

For example, statistics of the inference execution queue, model runtime parameters etc. They can also track the usage based on model version, API type or requested endpoint methods.

OpenVINO Model Server metrics are compatible with Prometheus standard [Prometheus metrics detailed description](https://prometheus.io/docs)

They are exposed on the `/metrics` endpoint.

## Available metrics families

Metrics from default list are enabled with the `metrics_enabled` flag or json configuration.

However, you can enable also additional metrics by listing all the metrics you want to enable in the `metric_list` flag or json configuration.


| Default metrics |
| Type      | Name | Labels | Description |
| :---    |    :----   |    :----   |    :----       |
| gauge      | ovms_streams | name,version | Number of OpenVINO execution streams |
| gauge      | ovms_current_requests | api,interface,method,name,version | Number of inference requests currently in process |
| counter      | ovms_requests_success | name,version | Number of successful requests to a model or a DAG. |
| counter      | ovms_requests_fail | name,version | Number of failed requests to a model or a DAG. |
| histogram      | ovms_request_time_us | interface,name,version | Processing time of requests to a model or a DAG. |
| histogram      | ovms_inference_time_us | name,version | Inference execution time in the OpenVINO backend. |
| histogram      | ovms_wait_for_infer_req_time_us | name,version | Request waiting time in the scheduling queue. |

| Optional metrics |
| Type      | Name | Labels | Description |
| :---    |    :----   |    :----   |    :----       |
| gauge      | ovms_infer_req_queue_size | name,version | Inference request queue size (nireq). |
| gauge      | ovms_infer_req_active | ,name,version | Number of currently consumed inference request from the processing queue. |

| Labels description |
| Name      | Values |  Description |
| :---    |    :----   |    :----   |
| api      | KServe, TensorFlowServing  | Name of the serving API. |
| interface      | REST, gRPC | Name of the serving interface. |
| method      | ModelMetadata, ModelReady, ModelInfer, Predict, GetModelStatus, GetModelMetadata | Interface methods. |
| version      | 1, 2, ..., n | Model version. Note that GetModelStatus and ModelReady do not have the version label. |
| name      | resnet, face_detection, ... | Model name. |


## Enable metrics

By default, the metrics feature is disabled.

Metrics endpoint is using the same port as the REST interface for running the model queries.

It is required to enable REST in the model server by setting the parameter --rest_port.

To enable default metrics set you need to specify the `metrics_enabled` flag or json setting:

CLI

   ```bash
         docker run --rm -d -v ${PWD}/models/resnet-50-tf:/opt/model -p 9001:9001 -p 9002:9002 openvino/model_server:latest \
               --model_path /opt/model --model_name resnet --port 9001 \
               --rest_port 3002 \
               --metrics_enabled
   ```

CONFIG CMD

   ```bash
         docker run --rm -d -v -d -v ${PWD}/workspace:/workspace openvino/model_server --config_path /workspace/config.json -p 9001:9001 -p 9002:9002 openvino/model_server:latest \
               --rest_port 9002
               --port 9001
   ```

CONFIG JSON

   ```bash
   {
    "model_config_list": [
        {
           "config": {
                "name": "resnet",
                "base_path": "/workspace/resnet-50-tf",
                "layout": "NHWC:NCHW",
                "shape": "(1,224,224,3)"
           }
        }
    ],
    "monitoring":
        {
            "metrics":
            {
                "enable" : true
            }
        }
   }
   ```

## Change the default list of metrics

You can enable from one up to all the metrics available at once.

To enable specific set of metrics you need to specify the metrics_list flag or json setting:

CLI

   ```bash
         docker run --rm -d -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
               --model_name resnet --model_path gs://ovms-public-eu/resnet50  --port 9000 \
               --rest_port 8000 \
               --metrics_enabled \
               --metrics_list ovms_requests_success,ovms_infer_req_queue_size
   ```

CONFIG CMD

   ```bash
         docker run --rm -d -v -d -v ${PWD}/workspace:/workspace openvino/model_server \
            --config_path /workspace/config.json -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
            --rest_port 8000 \
            --port 9000
   ```

CONFIG JSON

   ```bash
   {
    "model_config_list": [
        {
           "config": {
                "name": "resnet",
                "base_path": "/workspace/resnet-50-tf",
                "layout": "NHWC:NCHW",
                "shape": "(1,224,224,3)"
           }
        }
    ],
    "monitoring":
        {
            "metrics":
            {
                "enable" : true,
                "metrics_list": ["ovms_requests_success", "ovms_infer_req_queue_size"]
            }
        }
    }
    ```

CONFIG JSON WITH ALL METRICS ENABLED

```bash
   {
    "model_config_list": [
        {
           "config": {
                "name": "resnet",
                "base_path": "/workspace/resnet-50-tf",
                "layout": "NHWC:NCHW",
                "shape": "(1,224,224,3)"
           }
        }
    ],
    "monitoring":
        {
            "metrics":
            {
                "enable" : true,
                "metrics_list": 
                    [ "ovms_requests_success",
                    "ovms_requests_fail",
                    "ovms_inference_time_us",
                    "ovms_wait_for_infer_req_time_us",
                    "ovms_request_time_us",
                    "ovms_current_requests",
                    "ovms_infer_req_active",
                    "ovms_streams",
                    "ovms_infer_req_queue_size"]
            }
        }
    }
    ```

## Example response from metrics endpoint

To use data from metrics endpoint you can use the curl command:
```bash
    curl http://localhost:8000/metrics
```
[Example metrcis output](./metrics_output.log)

## Metrics implementation for DAG pipelines

For [DAG pipeline](dag_scheduler.md) execution there are relevant 3 metrics listed below.
They track the execution as a whole request processing all the pipeline nodes. 

| DAG metrics |
| Type      | Name  | Description |
| :---    |    :----   |    :----   |
| counter |    ovms_requests_success  |             Number of successful requests to a model or a DAG. |
| counter  |   ovms_requests_fail    |              Number of failed requests to a model or a DAG. |
| histogram |  ovms_request_time_us |               Processing time of requests to a model or a DAG. |

The remaining metrics track the execution for the individual models in the pipeline separately.
It means that each request to the DAG pipeline will update also the metrics for all individual models used as the execution nodes.
