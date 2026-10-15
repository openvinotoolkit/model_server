---
id: ovms_docs_server_app
sidebar_position: 1
---

# Write a Client Application

:::info Phase 1 structure note (pending stakeholder review)
Using-the-models landing/API chooser.
:::


OpenVINO&trade; Model Server supports multiple APIs, for easy integration with systems using one of them for inference.
The APIs are:

* KServe API for inference
* OpenAI API for text generation.
* OpenAI API for embeddings
* Cohere API for reranking

KServe API works on both gRPC and REST interfaces.
The REST API endpoints for generative use cases support both streamed and unary responses.

Check the following articles to learn more about the supported APIs:

- [KServe gRPC API](grpc-api-kserve.md)
- [KServe REST API](rest-api-kserve.md)
- [OpenAI chat completions API](rest-api-chat.md)
- [OpenAI completions API](rest-api-completions.md)
- [OpenAI embeddings API](rest-api-embeddings.md)
- [Cohere rerank API](rest-api-rerank.md)
- [OpenAI images generations API](rest-api-image-generation.md)
- [OpenAI speech recognition API](rest-api-speech-to-text.md)
- [OpenAI speech generation API](rest-api-text-to-speech.md)

In this section you can find short code samples to interact with OpenVINO Model Server endpoints via:
- [KServe API](kserve-api-index.md)
- [Generative AI clients](genai-api-index.md)
