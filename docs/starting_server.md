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

## Single model mode

The simplest way to deploy OpenVINO Model Server is in single model mode. In this mode only one model is served and the whole configuration is passed via CLI parameters.
>Note: In single model mode changing configuration in runtime is not available.

[Learn more](single_model_mode.md)

## Multi model mode with a configuration file

For serving multiple models use multi model mode. It requires configuration file that holds configuration for all served models. In this mode you can add and delete models as well as update their configuration in runtime, without restarting model server.

[Learn more](multiple_models_mode.md)