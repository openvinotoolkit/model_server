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

## Configuring Deployment
Depending on performance requirements, traffic projections, and  models, you may want to adjust the following:  

Server configuration options:
- ports used
- enable/disable REST API
- set configuration monitoring 

Configuration for each served models:  
- device where model is loaded
- model version policy
- inference related options

See the [model server parameters](parameters.md) page for additional details about the model server configuration. 

## Using AI Accelerators
Learn how to configure AI accelerators, including the Intel® Movidius™ Myriad™ X VPU, Intel® Data Center GPU Flex, and Intel® Movidius™ Myriad™ X High Density Deep Learning (HDDL), as well as Multi-Device (MULTI), Heterogeneous (HETERO) and AUTO device plugins for inference execution. 

[Learn more](accelerators.md)

## Keeping Models in Remote Storage
Leverage remote storage for your model repository. Use Google Cloud Storage (GCS), Amazon S3, Azure Blob Storage or any other S3-compatible storage (i.e., MinIO) to create more flexible model repositories which are easy to use and manage, for example, in Kubernetes deployments. 

[Learn more](using_cloud_storage.md)

## Keeping Deployments Secure
Security is an important consideration in model serving deployments. Ensure that appropriate permissions are in place to keep your models secure. Consider configuring access restrictions and traffic encryption to secure communication between clients and the model server. 

[Learn more](security_considerations.md)
