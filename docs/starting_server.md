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

## Serving a Single Model

The simplest way to deploy OpenVINO Model Server is with a single model. Only one model is served and the whole configuration is passed via command line parameters.
>NOTE: In single model mode changing configuration at runtime is not possible.

[Learn more](single_model_mode.md)

## Serving Multiple Models

To serve multiple models, use multi-model mode. This requires a configuration file that stores the configuration of all served models. In this mode you can add or delete models as well as update their configurations at runtime, without restarting Model Server.

[Learn more](multiple_models_mode.md)

## Run in a Container

OpenVINO Model Server is distributed as a Docker container image available from [DockerHub](https://hub.docker.com/r/openvino/model_server) and the [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de). Model Server images are minimal and contain only the necessary dependencies for inference serving. Using Docker is the recommended way to deploy OpenVINO Model Server.

[Learn more](docker_container.md)

## Run on Bare Metal and Virtual Machine (VM) Hosts

OpenVINO Model Server is open-source and implemented in C++. Therefore it's possible to download and compile the source code to obtain a binary and run on bare metal. The provided `make` targets simplify the build process.

[Learn more](host.md)

## Configure Deployments

Depending on which models are served, and what performance requirements and traffic expectations exist, you may want to configure some server options like:
- ports used
- enable/disable REST API
- set configuration monitoring 

... as well as the best fitting configuration for each of the served models like:
- which device to load the model onto
- the model version policy
- inference related options

[Learn more](parameters.md)

## Save Models in Remote Storage

Leverage remote storage options including Azure Blob Storage, Google Cloud Storage or any S3-compatible storage to create more elastic model repositories that are easy to use and manage in Kubernetes, for example.

[Learn more](using_cloud_storage.md)

## Set Model Versioning Policies

Take advantage of the model repository structure. Add or delete version directories and Model Server will automatically adjust what is being served. Take full control over served model versions by setting a model version policy and serving all, specific models or only the latest version of a model.

[Learn more](model_version_policy.md)

## Modify Model Configuration at Runtime

OpenVINO Model Server tracks changes to the configuration file and applies them during runtime. This means you can change model configurations (for example serve a model on a different device), add a new model or completely remove one that is no longer needed. Any changes will be applied with no disruptions of service functionality (no restart required).

[Learn more](online_config_changes.md)

## Secure Deployments

While deploying model server, think about security of your deployment. Take care of appropriate permissions and keeping your models in a safe place. Consider configuring access restrictions and traffic encryption to secure communication with the model server.

[Learn more](security_considerations.md)