# metrics {#ovms_docs_metrics}

## Introduction

This document describes how to use metrics endpoint in the OpenVINO Model Server. They can be applied for:

- Providing performance and utilization statistics for monitoring and benchmarking purposes

- Auto scaling of the model server instances in Kubernetes and OpenShift based on application related metrics

> **NOTE**: currently, Metrics feature is released as a preview feature.

Built-in metrics allows tracking the performance without any extra logic on the client side or using network traffic monitoring tools like load balancers or reverse-proxies.

It also exposes metrics which are not related to the traffic. 

For example statistics of the inference execution queue, model runtime parameters etc. They can also track the usage based on model version, API type or requested endpoint methods.

There are four types of metrics: 

Counter – counts the number of specific events 

Gauge – shows current state of the ovms or model 

Histograms – shows distributions of counters per range 

Summary – shows quantiles – will not be used on the server side 

For more information please visit [Prometheus metrics detailed description](https://prometheus.io/docs/concepts/metric_types/)

## Enable metrics

By default, the metrics feature is disabled.

To enable default metrics set you need to specify the metrics_enabled flag or json setting:

CLI

   ```bash
         docker run --rm -d -v ${PWD}/models/resnet-50-tf:/opt/model -p 9001:9001 -p 9002:9002 openvino/model_server:latest \
               --model_path /opt/model --model_name resnet --port 9001 \
               --rest_port 9002 \
               --metrics_enabled
   ```

CONFIG CMD

   ```bash
         docker run --rm -d -v -d -v ${PWD}/workspace:/workspace openvino/model_server --config_path /workspace/config.json -p 9001:9001 -p 9002:9002 openvino/model_server:latest \
               --rest_port 9002
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
         docker run --rm -d -v ${PWD}/models/resnet-50-tf:/opt/model -p 9001:9001 -p 9002:9002 openvino/model_server:latest \
               --model_path /opt/model --model_name resnet --port 9001 \
               --rest_port 9002 \
               --metrics_enabled
               --metrics_list ovms_requests_success,ovms_infer_req_queue_size
   ```

CONFIG CMD

   ```bash
         docker run --rm -d -v -d -v ${PWD}/workspace:/workspace openvino/model_server --config_path /workspace/config.json -p 9001:9001 -p 9002:9002 openvino/model_server:latest \
               --rest_port 9002
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

## Available metrics families

Metrics from default list are enabled with the metrics_enabled flag or json configuration.

However, you can enable also additional metrics by listing all the metrics you want to enable in the metric_list flag or json configuration.

DEFAULT

   ```bash
        ovms_streams
        ovms_current_requests
        ovms_requests_success
        ovms_requests_fail
        ovms_request_time_us
        ovms_inference_time_us
        ovms_wait_for_infer_req_time_us
   ```

ADDITIONAL

   ```bash
        ovms_infer_req_queue_size
        ovms_infer_req_active
   ```

## List of available metrics per interface

Metrics from default list are enabled with the metrics_enabled flag or json configuration.

However, you can enable also additional metrics by listing all the metrics you want to enable in the metric_list flag or json configuration.

ovms_requests_success

   ```bash
        ovms_requests_success_grpc_predict
        ovms_requests_success_grpc_getmodelmetadata
        ovms_requests_success_grpc_getmodelstatus
        ovms_requests_success_grpc_modelinfer
        ovms_requests_success_grpc_modelmetadata
        ovms_requests_success_grpc_modelready
        ovms_requests_success_rest_modelinfer
        ovms_requests_success_rest_predict
        ovms_requests_success_rest_modelmetadata
        ovms_requests_success_rest_modelstatus
        ovms_requests_success_rest_modelmetadata
        ovms_requests_success_rest_modelready
   ```

ovms_requests_fail

   ```bash
        ovms_requests_fail_grpc_predict
        ovms_requests_fail_grpc_getmodelmetadata
        ovms_requests_fail_grpc_getmodelstatus
        ovms_requests_fail_grpc_modelinfer
        ovms_requests_fail_grpc_modelmetadata
        ovms_requests_fail_grpc_modelready
        ovms_requests_fail_rest_modelinfer
        ovms_requests_fail_rest_predict
        ovms_requests_fail_rest_modelmetadata
        ovms_requests_fail_rest_modelstatus
        ovms_requests_fail_rest_modelmetadata
        ovms_requests_fail_rest_modelready
   ```

ovms_request_time_us

   ```bash
        ovms_request_time_us_grpc
        ovms_request_time_us_rest
   ```

## Changing metric endpoint port via CLI

Currently the metrics endpoint is located on the '''rest_port/metrics''' endpoint.

Changing the default setting is not part of the preview feature.

## Example response from metrics endpoint

TODO

## Example graphics from Graphana

TODO

## Alternative for internal OVMS metrics: 

Using 3rd party components acting as proxy between the clients and the server. 

Examples are HAproxy, IstIo mesh, openshift mesh, Kubernetes ingress controller. 

## Hardware specific metrics

HW specific metrics are considered out of scope for OVMS. They are typically handled by device specific metrics exporter.

They require specific knowledge about the HW drivers and monitoring API and often require privileged security mode which is not needed just for the inference execution.
