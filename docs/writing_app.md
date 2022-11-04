# Writing Client Application {#ovms_docs_server_app}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_server_api
   ovms_docs_clients

@endsphinxdirective

## Introduction

OpenVINO&trade; Model Server exposes two sets of APIs: one compatible with TensorFlow Serving and another one, with KServe API, for inference. Both APIs work on both gRPC and REST interfaces. Supporting two sets of APIs makes OpenVINO Model Server easier to plug into existing systems the already leverage one of those APIs for inference. Learn more about supported APIs:

- [TensorFlow Serving gRPC API](./model_server_grpc_api_tfs.md)
- [KServe gRPC API](./model_server_grpc_api_kfs.md)
- [TensorFlow Serving REST API](./model_server_rest_api_tfs.md)
- [KServe REST API](./model_server_rest_api_kfs.md)

If you already use one of these APIs, integration of OpenVINO Model Server should be smooth and transparent.