# Write a Client Application {#ovms_docs_server_app}

```{toctree}
---
maxdepth: 1
hidden:
---

OpenAI API <ovms_docs_clients_openai>
TensorFlow Serving API <ovms_docs_clients_tfs>
KServe API <ovms_docs_clients_kfs>
OpenVINO Model Server C-API <ovms_docs_c_api>
```

OpenVINO&trade; Model Server exposes three sets of APIs: one compatible with TensorFlow Serving, second with KServe API for inference and OpenAPI API for text generation. Both TFS and KServe APIs work on gRPC and REST interfaces.
OpenAI API `chat/completion` endpoint supports REST API calls with and without streamed responses.
 Supporting multiple APIs makes OpenVINO Model Server easier to plug into existing systems the already leverage one of these APIs for inference. Learn more about supported APIs:

- [TensorFlow Serving gRPC API](./model_server_grpc_api_tfs.md)
- [KServe gRPC API](./model_server_grpc_api_kfs.md)
- [TensorFlow Serving REST API](./model_server_rest_api_tfs.md)
- [KServe REST API](./model_server_rest_api_kfs.md)
- [OpenAI chat completions API](./model_server_rest_api_chat.md)
- [OpenAI completions API](./model_server_rest_api_completion.md)

In this section you can find short code samples to interact with OpenVINO Model Server endpoints via:
- [TensorFlow Serving API](./clients_tfs.md)
- [KServe API](./clients_kfs.md)
- [OpenAI API](./clients_openai.md)
