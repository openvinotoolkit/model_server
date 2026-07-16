# Write a Client Application {#ovms_docs_server_app}

```{toctree}
---
maxdepth: 1
hidden:
---

Generative AI Use Cases <ovms_docs_clients_genai>
KServe API <ovms_docs_clients_kfs>
OpenVINO Model Server C-API <ovms_docs_c_api>
```

OpenVINO&trade; Model Server supports multiple APIs, for easy integration with systems using one of them for inference.
The APIs are:

* KServe API for inference
* OpenAI API for text generation.
* OpenAI API for embeddings
* Cohere API for reranking

KServe API works on both gRPC and REST interfaces.
The REST API endpoints for generative use cases support both streamed and unary responses.

Check the following articles to learn more about the supported APIs:

- [KServe gRPC API](./model_server_grpc_api_kfs.md)
- [KServe REST API](./model_server_rest_api_kfs.md)
- [OpenAI chat completions API](./model_server_rest_api_chat.md)
- [OpenAI completions API](./model_server_rest_api_completions.md)
- [OpenAI embeddings API](./model_server_rest_api_embeddings.md)
- [Cohere rerank API](./model_server_rest_api_rerank.md)
- [OpenAI images generations API](./model_server_rest_api_image_generation.md)
- [OpenAI speech recognition API](./model_server_rest_api_speech_to_text.md)
- [OpenAI speech generation API](./model_server_rest_api_text_to_speech.md)

In this section you can find short code samples to interact with OpenVINO Model Server endpoints via:
- [KServe API](./clients_kfs.md)
- [Generative AI clients](./clients_genai.md)
