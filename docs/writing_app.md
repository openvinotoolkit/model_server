# Writing a Client Application {#ovms_docs_server_app}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_clients
   ovms_docs_server_api

@endsphinxdirective

OpenVINO&trade; Model Server exposes two sets of APIs: one compatible with TensorFlow Serving and another one, with KServe API, for inference. Both APIs work on gRPC and REST interfaces. Supporting two sets of APIs makes OpenVINO Model Server easier to plug into existing systems the already leverage one of these APIs for inference. Learn more about supported APIs:

- [TensorFlow Serving gRPC API](./model_server_grpc_api_tfs.md)
- [KServe gRPC API](./model_server_grpc_api_kfs.md)
- [TensorFlow Serving REST API](./model_server_rest_api_tfs.md)
- [KServe REST API](./model_server_rest_api_kfs.md)

In this section you can find short code samples to interact with OpenVINO Model Server endpoints via:
- [TensorFlow Serving API](./clients_tfs.md)
- [KServe API](./clients_kfs.md)
