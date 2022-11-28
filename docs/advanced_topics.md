# Advanced Features {#ovms_docs_advanced}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_sample_cpu_extension
   ovms_docs_model_cache
   ovms_docs_custom_loader

@endsphinxdirective

<br>[CPU Extensions](/src/example/SampleCpuExtension/README.md)<br>
Implement any CPU layer, unsupported by OpenVINO, as a shared library.

<br>[Model Cache](model_cache.md)<br>
Leverage an [OpenVINO&trade; model cache functionality](https://docs.openvino.ai/2022.2/openvino_docs_IE_DG_Model_caching_overview.html) to speed up subsequent model loading on a target device.

<br>[Custom Model Loader](custom_model_loader.md)<br>
Write your own custom model loader based on the predefine interface and load the same as a dynamic library. 