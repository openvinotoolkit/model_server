# Starting the Server {#ovms_docs_starting_server}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_single_model
   ovms_docs_multiple_models
   ovms_docs_docker_container
   ovms_docs_baremetal
   ovms_docs_parameters
   ovms_docs_cloud_storage
   ovms_docs_target_devices
   ovms_docs_model_version_policy
   ovms_docs_online_config_changes
   ovms_docs_security


@endsphinxdirective

## Serve single model

The simplest way to deploy OpenVINO Model Server is in single model mode. In this mode only one model is served and the whole configuration is passed via CLI parameters.
>Note: In single model mode changing configuration in runtime is not available.

[Learn more](single_model_mode.md)

## Serve multiple models

For serving multiple models use multi model mode. It requires configuration file that holds configuration for all served models. In this mode you can add and delete models as well as update their configuration in runtime, without restarting model server.

[Learn more](multiple_models_mode.md)

## Run in Docker container

OpenVINO Model Server is distributed as a Docker image and is available on [DockerHub](https://hub.docker.com/r/openvino/model_server) and [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de). Model server images are minimal and contain only necessary dependencies. Using Docker is recommended way of running OpenVINO Model Server.

[Learn more](docker_container.md)

## Run on bare metal and virtual hosts

OpenVINO Model Server is an open source project written in C++. Therefore it's possible to download and compile the code to obtain the binary and run it on bare metal. There are `make` targets provided to make this process simpler.

[Learn more](host.md)

## Configure deployment to fit your needs

Depending on what models OVMS serves, what are performance requirements and traffic expectations, you may want to configure some server options like:
- used ports
- enable/disable REST API
- set configuration monitoring 

... as well as best fitting configuration for each of the served models like:
- device to load the model onto
- model version policy
- inference related options

[Learn more](parameters.md)

## Hold your models in remote storage

Leverage remote storages such as Azure Blob Storage, Google Cloud Storage or AWS S3 compatible to create more elastic model repositories, that are easy to use and manage for example in Kubernetes deployments.

[Learn more](using_cloud_storage.md)

## Set up model versioning policies for served models

Take adventage of the model repository stucture. Add or delete version directories and model server will automatically adjust what is actually served. Have full control over served model versions by setting model version policy and serving all, only specific or always the latest version of the model.

[Learn more](model_version_policy.md)

## Modify models configuration in runtime with no service disruptions

OpenVINO Model Server tracks changes in the configuration file and applies them in runtime. It means that you can change models configuration (for example serve model on different device), add new model or completely remove one that is not needed anymore. Any changes will be applied with no disruptions of service functionality (no restart required).

[Learn more](online_config_changes.md)

## Keep your deployments secure

While deploying model server, think about security of your deployment. Take care of appropriate permissions and keeping your models in a safe place. Consider configuring access restrictions and traffic encryption to secure communication with the model server.

[Learn more](security_considerations.md)