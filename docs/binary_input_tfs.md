# Predict on Binary Inputs via TensorFlow Serving API {#ovms_docs_binary_input_tfs}

## GRPC

TensorFlow Serving API allows sending the model input data in a variety of formats inside the [TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) objects.
Array data is passed inside the `tensor_content` field, which represents the input data buffer.

When the data is sent in the `string_val` field to the model or pipeline that have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions, such input is interpreted as a binary encoded image. 

Note, that while the model metadata reports the inputs shape with layout NHWC, the binary data must be sent with 
shape: [N] with dtype: DT_STRING. Where N represents number of images converted to string bytes.

When sending data in the array format, all bytes are in the same sequence in `tensor_content` field and when loaded on the server side, the shape gives information on how to interpret them. For binary encoded data, bytes for each image in the batch are put in a separate sequence in the `string_val` field. The only information given by the `tensor_shape` field is the amount of images in the batch. On the server side, the bytes in each element of the `string_val` field are loaded, resized to match model input shape and converted to the OpenVINO-friendly array format by OpenCV.

## HTTP

TensorFlow Serving API also allows sending encoded images via HTTP interface to the model or pipeline that have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions. The binary data needs to be Base64 encoded and put into `inputs` or `instances` field as a map in form:

```
<input_name>: {"b64":<Base64 encoded data>}
```

On the server side, the Base64 encoded data is decoded to raw binary and loaded using OpenCV which then converts it to OpenVINO-friendly data format for inference.
   
## API Reference
- [TensorFlow Serving gRPC API Reference Guide](./model_server_grpc_api_tfs.md)
- [TensorFlow Serving REST API Reference Guide](./model_server_rest_api_tfs.md)

## Usage examples

Sample clients that use binary inputs via TFS API can be found here ([REST sample](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/ovmsclient/samples/http_predict_binary_resnet.py))/([GRPC sample](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/ovmsclient/samples/grpc_predict_binary_resnet.py))
Also, see the ([README](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/ovmsclient/samples/README.md))


## Recommendations:

Sending the data in binary format can significantly simplify the client code and it's preprocessing load. With the REST API
client, only curl and base64 tool or the requests python package is needed. In case the original input data is jpeg or png 
encoded, there is no preprocessing needed to send the request.

Binary data can significantly reduce the network utilization. In many cases it allows reducing the latency and achieve
very high throughput even with slower network bandwidth.
