# Online Configuration Updates {#ovms_docs_online_config_changes}

### Updating Configuration File
OpenVINO Model Server monitors changes to the configuration file and applies required modifications during runtime using two different methods:

1. Automatically, with an interval defined by the parameter `--file_system_poll_wait_seconds`. (introduced in version 2021.1)

2. On demand, using the [Config Reload API](./model_server_rest_api.md). (introduced in version 2021.3)

Configuration reload triggers the following operations:

- new model(s) or [DAG(s)](./dag_scheduler.md) are added to the configuration file, loaded and served.
- changes to the configured model storage (e.g. new model version is added) are applied. 
- changes to the configuration of deployed models and [DAGs](./dag_scheduler.md) are applied. 
- all model versions will be reloaded when there is a change to the model configuration.
- when a deployed model or [DAG](./dag_scheduler.md) is deleted from `config.json`, it will be unloaded completely from the server after already running inference operations have been completed.
- [DAGs](./dag_scheduler.md) that depend on changed or removed models are reloaded.
- changes to [custom loaders](./custom_model_loader.md) and custom node library configs are applied.

Model Server behavior in case of errors during configuration reloading:

- if a new `config.json` is not compliant with JSON schema, no changes are applied to the served models.
- if the new model, [DAG](./dag_scheduler.md) or [custom loader](./custom_model_loader.md) has an invalid configuration, it will be ignored until the next configuration reload. Configurations may be invalid due to incorrect paths (leading to non-existent directories), forbidden values in the config, invalid [DAG](./dag_scheduler.md) structure (e.g. cycle found in a graph), etc.
- an error occurs when a model, [DAG](./dag_scheduler.md) or [custom loader](./custom_model_loader.md) is reloading but does not prevent the reload of the remaining updated models.
- errors from configuration reload are triggered internally and saved in the logs. If [Config Reload API](./model_server_rest_api.md) is used, the response will also contain an error message. 

