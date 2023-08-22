# Model Server Features {#ovms_docs_features}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_dag
   ovms_docs_binary_input
   ovms_docs_text
   ovms_docs_model_version_policy
   ovms_docs_shape_batch_layout
   ovms_docs_online_config_changes
   ovms_docs_stateful_models
   ovms_docs_metrics
   ovms_docs_dynamic_input
   ovms_docs_c_api
   ovms_docs_advanced
   ovms_docs_mediapipe

@endsphinxdirective

## Serving Pipelines of Models
Connect multiple models in a pipeline and reduce data transfer overhead with Directed Acyclic Graph (DAG) Scheduler. 
Implement model inference and data transformations using a custom node C/C++ dynamic library.

[Learn more](dag_scheduler.md)

## Processing Raw Data
Send data in JPEG or PNG formats to reduce traffic and offload data pre-processing to the server.

[Learn more](binary_input.md)

## Model Versioning Policies
The model repository structure enables adding or deleting numerical version directories and the server will automatically adjust which models are served.  
Control which model versions are served by setting the model version policy to serve all models, a specific model or set of models or just the latest version of the model (default setting).

[Learn more](model_version_policy.md)

## Model Reshaping
Change the batch size, shape and layout of the model at runtime to achieve high throughput and low latency.

[Learn more](shape_batch_size_and_layout.md)

## Modify Model Configuration at Runtime
OpenVINO Model Server regularly checks for changes to the configuration file and applies them during runtime. This means that you can change model configurations 
(for example, change the device where a model is served), add a new model or completely remove one that is no longer needed. These changes will be applied without any disruption to the service.

[Learn more](online_config_changes.md)

## Working with Stateful Models
Serve models that operate on sequences of data and maintain their state between inference requests.

[Learn more](stateful_models.md)

## Metrics
Use the metrics endpoint compatible with the Prometheus to access performance and utilization statistics.

[Learn more](metrics.md)

## Enabling Dynamic Inputs
Configure served models to accept data with variable batch sizes and input shapes.

[Learn more](dynamic_input.md)

## Model Server C API
Use in process inference via model server to leverage the model management and model pipelines functionality of OpenVINO Model Server within an application. This allows to reuse existing OVMS functionality to execute inference locally without network overhead.

[Learn more](model_server_c_api.md)

## Advanced Features
Use CPU Extensions, model cache feature or a custom model loader.

[Learn more](advanced_topics.md)
