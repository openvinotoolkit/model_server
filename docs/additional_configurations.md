# Additional Configurations {#ovms_docs_additional_configurations}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_parameters
   ovms_docs_target_devices
   ovms_docs_cloud_storage
   ovms_docs_security

@endsphinxdirective

[Configuring Deployment](parameters.md)\
Depending on performance requirements, traffic expectations, and  models, you may want to make certain adjustments to:  

configuration of server options:
- ports used
- enable/disable REST API
- set configuration monitoring 

configuration for each of the served models:  
- the device to load the model onto
- the model version policy
- inference related options

Read about the [model server parameters](parameters.md) to get more details on the model server configuration. 

[Using AI Accelerators](accelerators.md)\
Learn how to configure AI accelerators, such as Intel Movidius Myriad VPUs, 
GPU, and HDDL, as well as Multi-Device, Heterogeneous and Auto Device Plugins for inference execution. 

[Keeping Models in a Remote Storage](using_cloud_storage.md)\
Leverage remote storages, compatible with Google Cloud Storage (GCS), Amazon S3, or Azure Blob Storage, to create more flexible model repositories 
that are easy to use and manage, for example, in Kubernetes deployments. 

[Keeping Deployments Secure](security_considerations.md)\
While deploying model server, think about security of your deployment. Take care of appropriate permissions and keeping your models in a safe place. 
Consider configuring access restrictions and traffic encryption to secure communication with the model server.
