# Predict on Binary Inputs via KServe API {#ovms_docs_binary_input_kfs}

## GRPC

KServe API allows sending the model input data in a variety of formats inside the [InferTensorContents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data-1) objects or in `raw_input_contents` field of [ModelInferRequest](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).
   
When the data is sent to the model or pipeline that have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions and input `datatype` is set to `BYTES`, such input is interpreted as a binary encoded image. Data of such inputs **must** be placed in `bytes_contents` or in `raw_input_contents` 

If data is located in `raw_input_contents` you need to precede data of every batch by 4 bytes(little endian) containing size of this batch. For example, if batch would contain three images of sizes 370, 480, 500 bytes the content of raw_input_contents[index_of_the_input] would look like this: 
<0x72010000 (=370)><370 bytes of first image><0xE0010000 (=480)><480 bytes of second image> <0xF4010000 (=500)><500 bytes of third image>

Note, that while the model metadata reports the inputs shape with layout `NHWC`, the binary data must be sent with 
shape: `[N]` with datatype: `BYTES`. Where `N` represents number of images converted to string bytes.

When sending data in the array format, the shape and datatype gives information on how to interpret bytes in the contents. For binary encoded data, the only information given by the `shape` field is the amount of images in the batch. On the server side, the bytes of every batch are loaded, resized to match model input shape and converted to the OpenVINO-friendly array format by OpenCV.

## HTTP

### JPEG / PNG encoded images

KServe API also allows sending encoded images via HTTP interface to the model or pipeline that have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions. Similar to GRPC input with such datatype `datatype` needs to be `BYTES`. The tensor binary data is provided in the request body, after JSON object. While the JSON part contains information required to route the data to the target model and run inference properly, the data itself, in the binary format is placed right after the JSON. Therefore, you need to precede data of every image by 4 bytes(little endian) containing size of this image and specify their combined size in `binary_data_size` parameter.

For binary inputs, the `parameters` map in the JSON part contains `binary_data_size` field for each binary input that indicates the size of the data on the input. Since there's no strict limitations on image resolution and format (as long as it can be loaded by OpenCV), images might be of different sizes. To send a batch of images you need to precede data of every batch by 4 bytes(little endian) containing size of this batch and specify their combined size in `binary_data_size`. For example, if batch would contain three images of sizes 370, 480, 500 bytes the content of input buffer inside binary extension would look like this: 
<0x72010000 (=370)><370 bytes of first image><0xE0010000 (=480)><480 bytes of second image> <0xF4010000 (=500)><500 bytes of third image>
And in that case binary_data_size would be 1350(370 + 480 + 500)
Function set_data_from_numpy in triton client lib that we use in our [REST sample](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/kserve-api/samples/http_infer_binary_resnet.py) automatically converts given images to this format.

If the request contains only one input `binary_data_size` parameter can be omitted - in this case whole buffer is treated as a input image.

For HTTP request headers, `Inference-Header-Content-Length` header must be provided to give the length of the JSON object, and `Content-Length` continues to give the full body length (as HTTP requires). See an extended example with the request headers, and multiple images in the batch:

On the server side, the binary encoded data is loaded using OpenCV which then converts it to OpenVINO-friendly data format for inference.

The structure of the response is specified [Inference Response specification](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-response-json-object).

### Raw data

Above section described how to send JPEG/PNG encoded image via REST interface. Data sent like this is processed by OpenCV to convert it to OpenVINO-friendly format. Many times data is already available in OpenVINO-friendly format and all you want to do is to send it and get the prediction.

With KServe API you can also send raw data in a binary representation via REST interface. **That way the request gets smaller and easier to process on the server side, therefore using this format is more efficient when working with RESTful API, than providing the input data in a JSON object**. To send raw data in the binary format, you need to specify `datatype` other than `BYTES` and data `shape`, should match the input `shape` (also the memory layout should be compatible). 

For the Raw Data binary inputs `binary_data_size` parameter can be omitted since the size of particular input can be calculated from its shape.

## API Reference

- [Kserve gRPC API Reference Guide](./model_server_grpc_api_kfs.md)
- [Kserve REST API Reference Guide](./model_server_rest_api_kfs.md)

## Usage examples

Sample clients that use binary inputs via KFS API can be found here ([REST sample](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/kserve-api/samples/http_infer_binary_resnet.py))/([GRPC sample](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/kserve-api/samples/grpc_infer_binary_resnet.py))
Also, see the ([README](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/kserve-api/samples/README.md))


## Recommendations:

Sending the data in binary format can significantly simplify the client code and it's preprocessing load. With the REST API
client, only curl or the requests python package is needed. In case the original input data is jpeg or png 
encoded, there is no preprocessing needed to send the request.

Binary data can significantly reduce the network utilization. In many cases it allows reducing the latency and achieve
very high throughput even with slower network bandwidth.
