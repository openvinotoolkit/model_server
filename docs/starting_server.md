# Starting the Server {#ovms_docs_starting_server}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_models_repository
   ovms_docs_docker_container
   ovms_docs_baremetal
   ovms_docs_single_model
   ovms_docs_multiple_models
   ovms_docs_parameters
   ovms_docs_cloud_storage
   ovms_docs_target_devices
   ovms_docs_kubernetes
   ovms_docs_security
   

@endsphinxdirective

## Serving a Single Model

The simplest way to deploy OpenVINO™ Model Server is in the single-model mode - only one model is served and the whole configuration is passed via CLI parameters.

> **NOTE**: In the single-model mode, changing configuration in runtime is not possible.

[Learn more](single_model_mode.md)

## Serving Multiple Models

To serve multiple models, use the multi-model mode. This requires a configuration file that stores settings for all served models. 
In this mode you can add and delete models, as well as update their configurations in runtime, without restarting Model Server.

[Learn more](multiple_models_mode.md)

## Running in a Docker Container

Using Docker is the recommended way of running OpenVINO Model Server. Its images are available via 
[DockerHub](https://hub.docker.com/r/openvino/model_server) and [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de). 
They are minimal and contain only the necessary dependencies.

[Learn more](docker_container.md)

## Running on Bare Metal and Virtual Machine (VM) Hosts

OpenVINO Model Server is an open-source project written in C++. Therefore you can download and compile the code to obtain the binary and run it on bare metal.
The `make` targets are provided to simplify the process.

[Learn more](host.md)

## Configuring Deployments

Depending on performance requirements, traffic expectations, and what models OVMS serves, you may want to make certain adjustments to:  
configuration of server options, like:
- ports used
- enable/disable REST API
- set configuration monitoring 

configuration for each of the served models, like:  
- the device to load the model onto
- the model version policy
- inference related options

[Learn more](parameters.md)

## Keeping Models in a Remote Storage

Leverage remote storages, compatible with Google Cloud Storage (GCS), Amazon S3, or Azure Blob Storage, to create more flexible model repositories 
that are easy to use and manage, for example, in Kubernetes deployments.

[Learn more](using_cloud_storage.md)


## Keeping Deployments Secure

While deploying model server, think about security of your deployment. Take care of appropriate permissions and keeping your models in a safe place. 
Consider configuring access restrictions and traffic encryption to secure communication with the model server.

[Learn more](security_considerations.md)
