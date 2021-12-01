# Online_config file updates {#ovms_docs_online_config_changes}

### Updating Configuration File
OpenVINO Model Server monitors the changes in its configuration and applies required modifications in runtime in two ways:

- Automatically, with an interval defined by the parameter --file_system_poll_wait_seconds. (introduced in release 2021.1)
- On demand, by using [Config Reload API](./model_server_rest_api.md#config-reload). (introduced in release 2021.3)

Configuration reload triggers the following operations:

- new model or [DAGs](./dag_scheduler.md) added to the configuration file will be loaded and served by OVMS.
- changes made in the configured model storage (e.g. new model version is added) will be applied. 
- changes in the configuration of deployed models and [DAGs](./dag_scheduler.md) will be applied. 
- all model version will be reloaded when there is a change in model configuration.
- when a deployed model, [DAG](./dag_scheduler.md) is deleted from config.json, it will be unloaded completely from OVMS after already started inference operations are completed.
- [DAGs](./dag_scheduler.md) that depends on changed or removed models will also be reloaded.
- changes in [custom loaders](./custom_model_loader.md) and custom node libraries configs will also be applied.

OVMS behavior in case of errors during config reloading:

- if the new config.json is not compliant with json schema, no changes will be applied to the served models.
- if the new model, [DAG](./dag_scheduler.md) or [custom loader](./custom_model_loader.md) has invalid configuration it will be ignored till next configuration reload. Configuration may be invalid because of invalid paths(leading to non-existing directories), forbidden values in config, invalid structure of [DAG](./dag_scheduler.md) (e.g. found cycle in a graph), etc.
- an error during one model reloading, [DAG](./dag_scheduler.md) or [custom loader](./custom_model_loader.md) does not prevent the reload of the remaining updated models.
- errors from configuration reloads triggered internally are saved in the logs. If [Config Reload API](./model_server_rest_api.md#config-reload) was used, also the response contains an error message. 

