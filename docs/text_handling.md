# Support for text data format {#ovms_docs_text}

OpenVINO Model Server can now greatly simplify writing the applications with Natural Language Processing models. For the use cases related to text analysis or text generation, the client application can communicate with the model server using the original text format. There is no requirement to perform pre and post processing on the client side. Tokenization and detokenization can be now fully delegated to the server.

We addressed both the situation when the original model requires tokens on input or output and there is added support for models with embedded tokenization layer. Below are demonstrated use cases with a simple client application sending and receiving text in a string format. Whole complexity of the text conversion is fully delegated to the remote serving endpoint.


## DAG pipeline to delegate tokenization to the server
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
There is a built-in [Tokenizer](https://github.com/openvinotoolkit/model_server/tree/develop/src/custom_nodes/tokenizer) custom node for that use case based on Blingfire library.

Similarly, a custom node can perform string detokenization and return a string to the model server client.

Check the [end-to-end demonstration](../demos/gptj_causal_lm/python/README.md) of such use case with GPT based text generation.

The client API snippets with string data format are included in [KServe API](./clients_kfs.md) and [TFS API](./clients_tfs.md).


## Custom CPU extension for tokenization layer in the model

Some AI model training frameworks supports the layers accepting the string format on the input or output. They include the layers performing the tokenization operations inside the neural network.
While OpenVINO doesn't support natively string data type, it is possible to extend the capabilities with a CPU extension.
We included in the model server a built-in extension for [SentencepieceTokenizer](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations) layer from TensorFlow.
The extension is capable of converting 1D U8 OpenVINO tensor into appropriate format for [SentencepieceTokenizer]. OVMS is able to detect such layer and create 1D U8 tensor out of KServe/TensorflowServing API strings automatically.

A demonstration of such use case is in the MUSE model which can be imported directly but the models server. The client can send the text data without any preprocessing and take advantage of much faster execution time.
Check the [MUSE demo](../demos/universal-sentence-encoder/README.md).



