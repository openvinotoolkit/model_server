# Support for text data format {#ovms_docs_text}

OpenVINO Model Server can now greatly simplify writing the applications with Natural Language Processing models. For the use cases related to text analysis or text generation, the client application can communicate with the model server using the original text format. There is no requirement to perform pre and post processing on the client side. Tokenization and detokenization can be now fully delegated to the server.

We addressed both the situation when the original model requires tokens on input or output and there is added support for models with embedded tokenization layer. Below are demonstrated use cases with a simple client application sending and receiving text in a string format. Whole complexity of the text conversion is fully delegated to the remote serving endpoint.

## Serving a MediaPipe graph with strings processing via a python node

When the model server is configured to serve python script (via MediaPipe Graph with PythonExecutorCalculator), it is possible to send a string or a list of strings. Refer to full [Python execution documentation](python_support/reference.md).

## Serving a model with a string in input or output layer

Some AI models support the layers with string format on the input or output. They include the layers performing the tokenization operations inside the neural network.

OpenVINO supports such layers with string data type using a CPU extension.
Model Server includes a built-in extension for wide list of custom [tokenizers and detokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) layers.
The extension performs tokenization operation for the string data type. 

A demonstration of such use case is in the MUSE model which can be imported directly but the models server. The client can send the text data without any preprocessing and take advantage of much faster execution time.
Check the [MUSE demo](../demos/universal-sentence-encoder/README.md).

Example usage of a model with string on output is our [image classification with string output demo](../demos/image_classification_with_string_output/README.md). The original model used in this demo returns probabilities but we are adding to the model postprocessing function which returns the most likely label name as a string.

## DAG pipeline to delegate tokenization to the server (deprecated)
When the model is using tokens on input or output, you can create a DAG pipeline which include custom nodes performing pre and post processing.
OpenVINO Model Server can accept the text data format on the gRPC and REST API interface and deserializes it to the 2D array of bytes, where each row represents single, null terminated sentence, padded with `\0` aligned to longest batch.

Example of batch size 2 of the string input - `abcd` and `ab`:
```
[
    'a', 'b', 'c', 'd', 0,
    'a', 'b',  0 ,  0 , 0
]
```
Such data in a tensor format can be passed to the custom node to perform the preprocessing like string tokenization. The output of the preprocessing node can be passed to the model.

Similarly, a custom node can perform string detokenization and return a string to the model server client.

The client API snippets with string data format are included in [KServe API](./clients_kfs.md) and [TFS API](./clients_tfs.md).
