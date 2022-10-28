# OpenVINO&trade; Model Server Features {#ovms_docs_features}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_model_version_policy
   ovms_docs_shape_batch_layout
   ovms_docs_online_config_changes
   ovms_docs_stateful_models
   ovms_docs_metrics
   ovms_docs_dynamic_input

@endsphinxdirective

## Setting Model Versioning Policies for served models

Take advantage of the model repository structure. Add or delete version directories and Model Server will automatically adjust. 
Take full control over the served model versions by setting a model version policy and serving all, the chosen, or just the latest version of the model.

[Learn more](model_version_policy.md)

## Modifying Model Configuration in Runtime

OpenVINO Model Server tracks changes to the configuration file and applies them in runtime. It means that you can change model configurations 
(for example serve the model on a different device), add a new model or completely remove one that is no longer needed. All changes will be applied with no 
disruption to the service and no restart will berequired.

[Learn more](online_config_changes.md)