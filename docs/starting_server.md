# Starting the Server {#ovms_docs_starting_server}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_models_repository
   ovms_docs_deploying_server
   ovms_docs_serving_model
   ovms_docs_additional_configurations
   

@endsphinxdirective

## Preparing Model for Serving

The models used by OpenVINO Model Server need to be stored locally or hosted remotely by object storage services. 
Learn how to [prepare your model for serving](models_repository.md). 

## Deploying the Server 

[Using Docker](docker_container.md) is the recommended way of running OpenVINO Model Server. The images are available via 
[DockerHub](https://hub.docker.com/r/openvino/model_server) and [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de). 

## Serving Models

[Serving a single model](single_model_mode.md) is the simplest way to deploy OpenVINO™ Model Server. Only one model is served and the whole configuration is passed via CLI parameters.
Note that changing configuration in runtime while serving a single model is not possible. Serving multiple models requires a configuration file that stores settings for all served models. 
You can add and delete models, as well as update their configurations in runtime, without restarting the model server.
