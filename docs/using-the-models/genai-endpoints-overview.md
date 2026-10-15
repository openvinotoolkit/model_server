---
id: ovms_docs_genai
sidebar_position: 5
---

# GenAI Endpoints

:::info Phase 1 structure note (pending stakeholder review)
GenAI endpoint capability overview, not a top-level landing page.
:::

OpenVINO Model Server allows extending the REST API interface to support arbitrary input format and execute arbitrary pipeline implemented as a MediaPipe graph.

The client makes a call to `http://server:port/v3/...` URL and post a JSON payload. The request is dispatched to the graph based on the `model` field in the JSON content.

It supports both the unary calls with a single response per request and streams which return a series of responses per request.

Use cases currently built-in in the model server:


## Text generation

The implementation is compatible with the OpenAI API for [chat/completions](rest-api-chat.md) and [completions](rest-api-completions.md).
It supports a wide range of text generation models from Hugging Face Hub.
Internally it employs continuous batching and paged attention algorithms for efficient execution both on CPU and GPU.

Learn more about the [LLM graph configuration](../serving-custom-pipelines/genai-pipeline-reference.md) and [exporting the models from Hugging Face for serving](../../demos/common/export_models/README.md).

Check also the demo of [text generation](../../demos/continuous_batching/README.md)


## Text embeddings

Text embeddings transform the semantic meaning of the text into a numerical vector. This operation is crucial for text searching and algorithms like RAG (Retrieval Augmented Generation).
The model server has built-in support for requests compatible with OpenAI AI and [embeddings endpoint](rest-api-embeddings.md).
It can run the execution both on CPU and GPU.

Check the [demo](../../demos/embeddings/README.md) how this endpoint can be used in practice. 


## Documents reranking

Reranking process is used to sort the list of documents based on relevance in the context of a query. Just like text generation and embeddings, it is essential element or RAG chains.
We implemented the [rerank API](rest-api-rerank.md) from Cohere.
Like the rest of the endpoints, rerank can run on CPU and GPU.

Check the [demo](../../demos/rerank/README.md) how this endpoint can be used.


## Image generation

The implementation is compatible with the OpenAI API for [images/generations](rest-api-image-generation.md).
It supports a wide range of image generation models from Hugging Face Hub.

Learn more about the [Image generation calculator](../serving-custom-pipelines/image-generation-pipeline-reference.md) and [exporting the models from Hugging Face for serving](../../demos/common/export_models/README.md).

Check also the demo of [image generation](../../demos/image_generation/README.md)

## Speech recognition

The implementation is compatible with the OpenAI API for [audio/transcriptions](rest-api-speech-to-text.md#transcription) and [audio/translations](rest-api-speech-to-text.md#translation).
It supports a wide range of speech recognition models from Hugging Face Hub.

Check also the demo of [transcription](../../demos/audio/README.md#transcription) and [translation](../../demos/audio/README.md#translation).

## Speech generation

The implementation is compatible with the OpenAI API for [audio/speech](rest-api-text-to-speech.md).
It supports Kokoro speech generation models, for example `Kokoro-82M-int8-ov` served from `OpenVINO/Kokoro-82M-int8-ov`.

Check also the demo of [speech generation](../../demos/audio/README.md#speech-generation).



