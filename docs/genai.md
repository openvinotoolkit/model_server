# GenAI Endpoints {#ovms_docs_genai}

OpenVINO Model Server allows extending the REST API interface to support arbitrary input format and execute arbitrary pipeline implemented as a MediaPipe graph.

The client makes a call to http://<server>:<port>/v3/... URL and post a JSON payload. The request is dispatched to the graph based on the `model` field in the JSON content.

It supports both the unary calls with a single response per request and streams which return a series of responses per request.

Use cases currently built-in in the model server:


## Text generation

The implementation is compatible with the OpenAI API for [chat/completions](./model_server_rest_api_chat.md) and [completions](./model_server_rest_api_completions.md).
It supports a wide range of text generation models from Hugging Face Hub.
Internally it employs continuous batching and paged attention algorithms for efficient execution both on CPU and GPU.

Learn more about the [LLM graph configuration](./llm/reference.md) and [exporting the models from Hugging Face for serving](../demos/common/export_models/README.md).

Check also the demo of [text generation](../demos/continuous_batching/README.md)


## Text embeddings

Text embeddings transform the semantic meaning of the text into a numerical vector. This operation is crucial for text searching and algorithms like RAG (Retrieval Augmented Generation).
The model server has built-in support for requests compatible with OpenAI AI and [embeddings endpoint](./model_server_rest_api_embeddings.md).
It can run the execution both on CPU and GPU.

Check the [demo](../demos/embeddings/README.md) how this endpoint can be used in practice. 


## Documents reranking

Reranking process is used to sort the list of documents based on relevance in the context of a query. Just like text generation and embeddings, it is essential element or RAG chains.
We implemented the [rerank API](./model_server_rest_api_rerank.md) from Cohere.
Like the rest of the endpoints, rerank can run on CPU and GPU.

Check the [demo](../demos/rerank/README.md) how this endpoint can be used.








