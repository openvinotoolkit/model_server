# Model Version Policy {#ovms_docs_model_version_policy}

> Note: It only concerns single models. DAG Pipelines and MediaPipe Graphs are not versioned.

The model version policy determines which versions of a model or models will be served by the OpenVINO Model Server. 
This parameter enables controlling memory consumption of the server and deciding which versions will be used regardless of what exists
in the model repository when the server is started. The `model_version_policy` parameter is optional. 
By default, the server serves only the latest version of a model. The accepted format for parameters using the CLI and in the configuration file is JSON.

Accepted values :
```
{"all": {}}

{"latest": { "num_versions": Integer}}

{"specific": { "versions": List }}
```
Examples:
```
{"latest": { "num_versions": 2 }} # server will serve only 2 latest versions of model

{"specific": { "versions": [1, 3] }} # server will serve only 1 and 3 versions of given model

{"all": {}} # server will serve all available versions of given model
```
## Updating Model Versions
- Served versions of models are updated online by monitoring file system changes in the model repository. OpenVINO Model Server adds new versions to the serving list when a new numerical subfolder containing model files is added. The default served version will be switched to the highest numbered subdirectory. 

- When the model version is deleted from the file system, it will become unavailable on the server and it will release RAM allocation. Updates in the deployed model version files will not be detected and they will not trigger changes in serving.

- By default model server is detecting new and deleted versions in 1-second intervals. The frequency can be changed by setting a parameter --file_system_poll_wait_seconds. If set to zero, updates will be disabled.

