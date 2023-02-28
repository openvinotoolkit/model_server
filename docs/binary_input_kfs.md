# Predict on Binary Inputs via KServe API {#ovms_docs_binary_input_kfs}

## GRPC

KServe API allows sending the model input data in a variety of formats inside the [InferTensorContents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data-1) objects or in `raw_input_contents` field of [ModelInferRequest](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).
   
When the data is sent in the `bytes_contents` field of `InferTensorContents` and input `datatype` is set to `BYTES`, such input is interpreted as a binary encoded image. The `BYTES` datatype is dedicated to binary encoded **images** and if it's set, the data **must** be placed in `bytes_contents` or in `raw_input_contents` if batch size is equal to 1.

Note, that while the model metadata reports the inputs shape with layout `NHWC`, the binary data must be sent with 
shape: `[N]` with datatype: `BYTES`. Where `N` represents number of images converted to string bytes.

When sending data in the array format, the shape and datatype gives information on how to interpret bytes in the contents. For binary encoded data, the only information given by the `shape` field is the amount of images in the batch. On the server side, the bytes in each element of the `bytes_contents` field are loaded, resized to match model input shape and converted to the OpenVINO-friendly array format by OpenCV.

## HTTP

### JPEG / PNG encoded images

KServe API also allows sending binary encoded data via HTTP interface. The tensor binary data is provided in the request body, after JSON object. While the JSON part contains information required to route the data to the target model and run inference properly, the data itself, in the binary format is placed right after the JSON. 

For binary inputs, the `parameters` map in the JSON part contains `binary_data_size` field for each binary input that indicates the size of the data on the input. Since there's no strict limitations on image resolution and format (as long as it can be loaded by OpenCV), images might be of different sizes. Therefore, to send a batch of different images, specify their sizes in `binary_data_size` field as a list with sizes of all images in the batch.
The list must be formed as a string, so for example, for 3 images in the batch, you may pass - `"9821,12302,7889"`.
If the request contains only one input `binary_data_size` parameter can be omitted - in this case whole buffer is treated as a input image.

For HTTP request headers, `Inference-Header-Content-Length` header must be provided to give the length of the JSON object, and `Content-Length` continues to give the full body length (as HTTP requires). See an extended example with the request headers, and multiple images in the batch:

On the server side, the binary encoded data is loaded using OpenCV which then converts it to OpenVINO-friendly data format for inference.

The structure of the response is specified [Inference Response specification](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-response-json-object).

### Raw data

Above section described how to send JPEG/PNG encoded image via REST interface. Data sent like this is processed by OpenCV to convert it to OpenVINO-friendly format. Many times data is already available in OpenVINO-friendly format and all you want to do is to send it and get the prediction.

With KServe API you can also send raw data in a binary representation via REST interface. **That way the request gets smaller and easier to process on the server side, therefore using this format is more effecient when working with RESTful API, than providing the input data in a JSON object**. To send raw data in the binary format, you need to specify `datatype` other than `BYTES` and data `shape`, should match the input `shape` (also the memory layout should be compatible). 

For the Raw Data binary inputs `binary_data_size` parameter can be omitted since the size of particular input can be calculated from its shape.

## API Reference

- [Kserve gRPC API Reference Guide](./model_server_grpc_api_kfs.md)
- [Kserve REST API Reference Guide](./model_server_rest_api_kfs.md)

## Usage examples

Sample clients that use binary inputs via KServe API can be found here ([REST sample](https://github.com/openvinotoolkit/model_server/blob/releases/2022/3/client/python/kserve-api/samples/http_infer_binary_resnet.py))/([GRPC sample](https://github.com/openvinotoolkit/model_server/blob/releases/2022/3/client/python/kserve-api/samples/http_infer_binary_resnet.py))
Also, see the ([README](https://github.com/openvinotoolkit/model_server/blob/releases/2022/3/client/python/kserve-api/samples/README.md))

## Recommendations:

Sending the data in binary format can significantly simplify the client code and it's preprocessing load. With the REST API
client, only curl or the requests python package is needed. In case the original input data is jpeg or png 
encoded, there is no preprocessing needed to send the request.

Binary data can significantly reduce the network utilization. In many cases it allows reducing the latency and achieve
very high throughput even with slower network bandwidth.
