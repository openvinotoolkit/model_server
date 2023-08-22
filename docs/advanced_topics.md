# Advanced Features {#ovms_docs_advanced}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_sample_cpu_extension
   ovms_docs_model_cache
   ovms_docs_custom_loader
   ovms_extras_nginx-mtls-auth-readme

@endsphinxdirective

## CPU Extensions
Implement any CPU layer, that is not support by OpenVINO yet, as a shared library.

[Learn more](../src/example/SampleCpuExtension/README.md)

## Model Cache
Leverage the OpenVINO [model caching](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Model_caching_overview.html) feature to speed up subsequent model loading on a target device.

[Learn more](model_cache.md)

## Custom Model Loader
Write your own custom model loader based on a predefined interface and load it similar to a dynamic library.  

[Learn more](custom_model_loader.md)

## Securing Model Server with NGINX
Protect network endpoints with traffic encryption and client authorization using a reverse proxy

[Learn more](../extras/nginx-mtls-auth/README.md)