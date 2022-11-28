# Advanced Features {#ovms_docs_advanced}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_sample_cpu_extension
   ovms_docs_model_cache
   ovms_docs_custom_loader

@endsphinxdirective

## CPU Extensions
Implement any CPU layer, unsupported by OpenVINO, as a shared library.

[Learn more](../src/example/SampleCpuExtension/README.md)

## Model Cache
Leverage an [OpenVINO&trade; model cache functionality](https://docs.openvino.ai/2022.2/openvino_docs_IE_DG_Model_caching_overview.html) to speed up subsequent model loading on a target device.

[Learn more](model_cache.md)

## Custom Model Loader
Write your own custom model loader based on the predefine interface and load the same as a dynamic library. 

[Learn more](custom_model_loader.md)