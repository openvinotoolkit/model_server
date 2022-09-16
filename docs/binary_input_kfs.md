# Request Prediction on Binary Encoded Image via Kserve API {#ovms_docs_binary_input_kfs}

## GRPC

KServe API allows sending the model input data in a variety of formats inside the [InferTensorContents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data-1) objects or in `raw_input_contents` field of [ModelInferRequest](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).
   
When the data is sent in the `bytes_contents` field of `InferTensorContents` and input `datatype` is set to `BYTES`, such input is interpreted as a binary encoded image.

Note, that while the model metadata reports the inputs shape with layout `NHWC`, the binary data must be sent with 
shape: `[N]` with datatype: `BYTES`. Where `N` represents number of images converted to string bytes.

Let's see how the ModelInferRequest object may look like if you decide to send the image:

1) as an array

   ```
   ModelInferRequest {
      model_name: "my_model"
      inputs: [
         {
         datatype: "FLOAT32"
         shape: [2, 300, 300, 3]
         raw_input_contents : [\x11\x02\ ... \x75\x0a]
         }
      ]
   }
   ```

2) as binary data

   ```
   ModelInferRequest {
      model_name: "my_model"
      inputs: [
         {
         datatype: "BYTES"
         shape: [2]
         contents.bytes_contents: [[\x31\x92\ ... \xaa\x4a], [\x00\x00\ ... \xff\xff]]
         }
      ]
   }
   ```

When sending data in the array format, the shape and datatype gives information on how to interpret bytes in the contents. For binary encoded data, the only information given by the `shape` field is the amount of images in the batch. On the server side, the bytes in each element of the `bytes_contents` field are loaded, resized to match model input shape and converted to the OpenVINO-friendly array format by OpenCV.

## HTTP

KServe API also allows sending binary encoded data via HTTP interface. The tensor binary data is provided in the request body, after JSON object. While the JSON part contains information required to route the data to the target model and run inference properly, the data itself, in the binary format is placed right after the JSON. See the simple example:

```
{
"model_name" : "my_model",
"inputs" : [
   {
      "name" : "model_input",
      "shape" : [ 1 ],
      "datatype" : "BYTES",
      "parameters" : {
         "binary_data_size" : "9472"
      }
   }
]
}
<9472 bytes of data for model_input tensor>
```

For binary inputs, the `parameters` map in the JSON part contains `binary_data_size` field for each binary input that indicates the size of the data on the input. Since there's no strict limitations on image resolution and format (as long as it can be loaded by OpenCV), images might be of different sizes. Therefore, to send a batch of different images, specify their sizes in `binary_data_size` field as a list with sizes of all images in the batch.
The list must be formed as a string, so for example, for 3 images in the batch, you may pass - `"9821,12302,7889"`

For HTTP request headers, `Inference-Header-Content-Length` header must be provided to give the length of the JSON object, and `Content-Length` continues to give the full body length (as HTTP requires). See an extended example with the request headers, and multiple images in the batch:

```
POST /v2/models/my_model/infer HTTP/1.1
Host: localhost:5000
Content-Type: application/octet-stream
Inference-Header-Content-Length: <xx>
Content-Length: <xx+(9821+12302+7889)>
{
"model_name" : "my_model",
"inputs" : [
   {
      "name" : "model_input",
      "shape" : [ 3 ],
      "datatype" : "BYTES",
      "parameters" : {
         "binary_data_size" : "9821,12302,7889"
      }
   },

]
}
<9821 bytes of the first image in the batch for model_input tensor>
<12302 bytes of the first image in the batch for model_input tensor>
<7889 bytes of the first image in the batch for model_input tensor>
```

On the server side, the binary encoded data is loaded using OpenCV which then converts it to OpenVINO-friendly data format for inference.

The structure of the response is specified [Inference Response specification](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-response-json-object).


## API Reference

- [Kserve gRPC API Reference Guide](./model_server_grpc_api_kfs.md)
- [Kserve REST API Reference Guide](./model_server_rest_api_kfs.md)

## Usage example with binary input

Examples below assumes OVMS has been started with ResNet50 binary model:

```bash
docker run -d --rm -p 8000:8000 -p 9000:9000 openvino/model_server:latest \
--model_name resnet --model_path gs://ovms-public-eu/resnet50-binary --layout NHWC:NCHW --plugin_config '{"CPU_THROUGHPUT_STREAMS": "1"}' \
--port 9000 --rest_port 8000
```

Prepare the client:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/kserve-api/samples
pip install -r requirements.txt
```

### Run the gRPC client sending the binary input:
```bash
python grpc_predict_binary_resnet.py --images_dir ../../../../demos/common/static/images --model_name resnet --service_url localhost:9000
Image ../../../../demos/common/static/images/magnetic_compass.jpeg has been classified as magnetic compass
Image ../../../../demos/common/static/images/pelican.jpeg has been classified as pelican
Image ../../../../demos/common/static/images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image ../../../../demos/common/static/images/snail.jpeg has been classified as snail
Image ../../../../demos/common/static/images/zebra.jpeg has been classified as zebra
Image ../../../../demos/common/static/images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image ../../../../demos/common/static/images/bee.jpeg has been classified as bee
Image ../../../../demos/common/static/images/peacock.jpeg has been classified as peacock
Image ../../../../demos/common/static/images/airliner.jpeg has been classified as airliner
Image ../../../../demos/common/static/images/golden_retriever.jpeg has been classified as golden retriever
```


### Run the REST client sending the binary input:
```bash
python http_predict_binary_resnet.py --images_dir ../../../../demos/common/static/images --model_name resnet --service_url localhost:8000
Image ../../../../demos/common/static/images/magnetic_compass.jpeg has been classified as magnetic compass
Image ../../../../demos/common/static/images/pelican.jpeg has been classified as pelican
Image ../../../../demos/common/static/images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image ../../../../demos/common/static/images/snail.jpeg has been classified as snail
Image ../../../../demos/common/static/images/zebra.jpeg has been classified as zebra
Image ../../../../demos/common/static/images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image ../../../../demos/common/static/images/bee.jpeg has been classified as bee
Image ../../../../demos/common/static/images/peacock.jpeg has been classified as peacock
Image ../../../../demos/common/static/images/airliner.jpeg has been classified as airliner
Image ../../../../demos/common/static/images/golden_retriever.jpeg has been classified as golden retriever
```

## Error handling:
In case the binary input can not be converted to the array of correct shape, an error status is returned:
- 400 - BAD_REQUEST for REST API
- 3 - INVALID_ARGUMENT for gRPC API


## Recommendations:

Sending the data in binary format can significantly simplify the client code and it's preprocessing load. With the REST API
client, only curl or the requests python package is needed. In case the original input data is jpeg or png 
encoded, there is no preprocessing needed to send the request.

Binary data can significantly reduce the network utilization. In many cases it allows reducing the latency and achieve
very high throughput even with slower network bandwidth.
