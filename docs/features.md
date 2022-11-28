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

[Serving Pipelines of Models](dag_scheduler.md)<br />
Connect multiple models to deploy complex processing solutions and reducing data transfer overhead with Directed Acyclic Graph (DAG) Scheduler. 
Implement model inference and data transformations with a custom node C/C++ dynamic library.

[Processing Raw Data](binary_input.md)<br />
Send data in JPEG or PNG formats to reduce traffic and offload the client applications.

[Setting Model Versioning Policies for served models](model_version_policy.md)<br />
Take advantage of the model repository structure. Add or delete version directories and Model Server will automatically adjust. 
Take full control over the served model versions by setting a model version policy and serving all, the chosen, or just the latest version of the model.


[Model Reshaping](shape_batch_size_and_layout.md)<br />
Change batch, shape and layout of the model in runtime for high-throughput and low-latency

[Modifying Model Configuration in Runtime](online_config_changes.md)<br />
OpenVINO Model Server tracks changes to the configuration file and applies them in runtime. It means that you can change model configurations 
(for example serve the model on a different device), add a new model or completely remove one that is no longer needed. All changes will be applied with no 
disruption to the service and no restart will berequired.

[Working with Stateful Models](stateful_models.md)<br />
Serve models that operate on sequences of data and maintain their state between inference requests.

[Metrics](metrics.md)<br />

Use metrics endpoint compatible with Prometheus standard to get performance and utilization statistics.

[Enabling Dynamic Inputs](dynamic_input.md)<br />
Configure served models to accept data with different batch sizes and in different shapes.

[Advanced Features](advanced_topics.md)<br />
Use CPU Extensions, model cache feature or a custom model loader.
