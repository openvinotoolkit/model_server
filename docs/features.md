# Model Server Features {#ovms_docs_features}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_dag
   ovms_docs_binary_input
   ovms_docs_model_version_policy
   ovms_docs_shape_batch_layout
   ovms_docs_online_config_changes
   ovms_docs_stateful_models
   ovms_docs_metrics
   ovms_docs_dynamic_input
   ovms_docs_advanced

@endsphinxdirective

## Serving Pipelines of Models

Connect multiple models to deploy complex processing solutions and reducing data transfer overhead with Directed Acyclic Graph (DAG) Scheduler. 
Implement model inference and data transformations with a custom node C/C++ dynamic library.
[Learn more](dag_scheduler.md)

## Processing Raw Data

Send data in JPEG or PNG formats to reduce traffic and offload the client applications.
[Learn more](binary_input.md)

## Setting Model Versioning Policies for served models

Take advantage of the model repository structure. Add or delete version directories and Model Server will automatically adjust. 
Take full control over the served model versions by setting a model version policy and serving all, the chosen, or just the latest version of the model.
[Learn more](model_version_policy.md)

## Model Reshaping 

Change batch, shape and layout of the model in runtime for high-throughput and low-latency
[Learn more](shape_batch_size_and_layout.md) 

## Modifying Model Configuration in Runtime

OpenVINO Model Server tracks changes to the configuration file and applies them in runtime. It means that you can change model configurations 
(for example serve the model on a different device), add a new model or completely remove one that is no longer needed. All changes will be applied with no 
disruption to the service and no restart will berequired.
[Learn more](online_config_changes.md)

## Working with Stateful Models

Serve models that operate on sequences of data and maintain their state between inference requests.
[Learn more](stateful_models.md)

## Metrics

Use metrics endpoint compatible with Prometheus standard to get performance and utilization statistics.
[Learn more](metrics.md)

## Enabling Dynamic Inputs

Configure served models to accept data with different batch sizes and in different shapes.
[Learn more](dynamic_input.md)

## Advanced Features

Use CPU Extensions, model cache feature or a custom model loader.
[Learn more](advanced_topics.md)