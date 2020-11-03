# Model Version Policy in OpenVINO&trade; Model Server

Model version policy makes it possible to decide which versions of model will be served by OVMS. 
This parameter allows you to control the memory consumption of the server and decide which versions will be used regardless of what is located 
under the path given when the server is started. model_version_policy parameter is optional. 
By default server serves only the latest version for the model. Accepted format for parameter in CLI and in config is json.

Accepted values :

{"all": {}}

{"latest": { "num_versions": Integer}}

{"specific": { "versions": List }}

Examples:

{"latest": { "num_versions":2 }} # server will serve only 2 latest versions of model

{"specific": { "versions":[1, 3] }} # server will serve only 1 and 3 versions of given model

{"all": {}} # server will serve all available versions of given model

## Updating model versions
- Served versions are updated online by monitoring file system changes in the model storage. OpenVINO Model Server will add new version to the serving list when new numerical subfolder with the model files is added. The default served version will be switched to the one with the highest number. 

- When the model version is deleted from the file system, it will become unavailable on the server and it will release RAM allocation. Updates in the deployed model version files will not be detected and they will not trigger changes in serving.

- By default model server is detecting new and deleted versions in 1 second intervals. The frequency can be changed by setting a parameter --file_system_poll_wait_seconds. If set to zero, updates will be disabled.

