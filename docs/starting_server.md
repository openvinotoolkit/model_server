# Starting the Server {#ovms_docs_starting_server}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_models_repository
   ovms_docs_docker_container
   ovms_docs_single_model
   ovms_docs_multiple_models
   ovms_docs_parameters
   ovms_docs_target_devices
   ovms_docs_security
   

@endsphinxdirective

## Preparing Model for Serving

The models used by OpenVINO Model Server need to be stored locally or hosted remotely by object storage services. 
Learn how to [prepare your model for serving](models_repository.md). 

Leverage remote storages, compatible with Google Cloud Storage (GCS), Amazon S3, or Azure Blob Storage, to create more flexible model repositories 
that are easy to use and manage, for example, in Kubernetes deployments. [Learn more](using_cloud_storage.md)

## Running in a Docker Container

[Using Docker](docker_container.md) is the recommended way of running OpenVINO Model Server. The images are available via 
[DockerHub](https://hub.docker.com/r/openvino/model_server) and [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de). 

## Running on Bare Metal and Virtual Machine Hosts

OpenVINO Model Server is an open-source project written in C++. You can download and compile the code to obtain the binary and [run it on bare metal](host.md).
The `make` targets are provided to simplify the process.

## Serving a Single Model

[Serving a single model](single_model_mode.md) is the simplest way to deploy OpenVINO™ Model Server. Only one model is served and the whole configuration is passed via CLI parameters.
Note that changing configuration in runtime while serving a single model is not possible.

## Serving Multiple Models

[Serving multiple models](multiple_models_mode.md) requires a configuration file that stores settings for all served models. 
You can add and delete models, as well as update their configurations in runtime, without restarting the model server.

## Configuring Deployment

Depending on performance requirements, traffic expectations, and  models, you may want to make certain adjustments to:  

configuration of server options:
- ports used
- enable/disable REST API
- set configuration monitoring 

configuration for each of the served models:  
- the device to load the model onto
- the model version policy
- inference related options

Read [model server parameters](parameters.md) to get more details on the model server configuration. 

## Using AI Accelerators

Learn how to [configure AI accelerators](accelerators.md), such as Intel Movidius Myriad VPUs, 
GPU, and HDDL, as well as Multi-Device, Heterogeneous and Auto Device Plugins for inference execution. 

## Keeping Deployments Secure

While deploying model server, think about security of your deployment. Take care of appropriate permissions and keeping your models in a safe place. 
Consider configuring access restrictions and traffic encryption to secure communication with the model server.
[Learn more](security_considerations.md)
