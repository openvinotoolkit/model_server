# Support for text data format {#ovms_docs_streaming_endpoints}

OpenVINO Model Server can now greatly simplify writing the applications with Natural Language Processing models. For the use cases related to text analysis or text generation, the client application can communicate with the model server using the original text format. There is no requirement to perform pre and post processing on the client side. Tokenization and detokenization can be now fully delegated to the server.

We addressed both the situation when the original model requires tokens on input or output and there is added support for models with embedded tokenization layer. Below are demonstrated use cases with a simple client application sending and receiving text in a string format. Whole complexity of the text conversion is fully delegated to the remote serving endpoint.
