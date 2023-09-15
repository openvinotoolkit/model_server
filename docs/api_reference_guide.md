# API Reference Guide {#ovms_docs_server_api}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_grpc_api_tfs
   ovms_docs_grpc_api_kfs
   ovms_docs_rest_api_tfs
   ovms_docs_rest_api_kfs
   ovms_docs_c_api

@endsphinxdirective

## Introduction

OpenVINO&trade; Model Server exposes two sets of network APIs for inference: one compatible with TensorFlow Serving and another one, with KServe API. Both APIs work on gRPC and REST interfaces. Supporting two sets of APIs makes OpenVINO Model Server easier to plug into existing systems the already leverage one of those APIs for inference. Learn more about supported APIs:

- [TensorFlow Serving gRPC API](./model_server_grpc_api_tfs.md)
- [KServe gRPC API](./model_server_grpc_api_kfs.md)
- [TensorFlow Serving REST API](./model_server_rest_api_tfs.md)
- [KServe REST API](./model_server_rest_api_kfs.md)

If you already use one of these APIs, integration of OpenVINO Model Server should be smooth and transparent.

Additionally OVMS provides in process inference with its C API:
- [OVMS C API](./model_server_c_api.md)
