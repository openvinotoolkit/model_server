# Predict on Binary Inputs via TensorFlow Serving API {#ovms_docs_binary_input_tfs}

## GRPC

TensorFlow Serving API allows sending the model input data in a variety of formats inside the [TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) objects.
Array data is passed inside the `tensor_content` field, which represents the input data buffer.

When the data is sent in the `string_val` field, it is interpreted as a binary format of the input data.

Note, that while the model metadata reports the inputs shape with layout NHWC, the binary data must be sent with 
shape: [N] with dtype: DT_STRING. Where N represents number of images converted to string bytes.

Let's see how the `TensorProto` object may look like if you decide to send the image:

1) as an array
   
   ```
   TensorProto {
      dtype: DT_FLOAT32
      tensor_shape: [2, 300, 300, 3]
      tensor_content: [\x11\x02\ ... \x75\x0a]
   }
   ```

2) as binary data

   ```
   TensorProto {
      dtype: DT_STRING
      tensor_shape: [2]
      string_val: [[\xff\xff ... \x66\xa0], [\x00\x00 ... \x13\x41]]
   }
   ```   

When sending data in the array format, all bytes are in the same sequence in `tensor_content` field and when loaded on the server side, the shape gives information on how to interpret them. For binary encoded data, bytes for each image in the batch are put in a separate sequence in the `string_val` field. The only information given by the `tensor_shape` field is the amount of images in the batch. On the server side, the bytes in each element of the `string_val` field are loaded, resized to match model input shape and converted to the OpenVINO-friendly array format by OpenCV.

## HTTP

TensorFlow Serving API also allows sending binary encoded data via HTTP interface. The binary data needs to be Base64 encoded and put into `inputs` or `instances` field as a map in form:

```
<input_name>: {"b64":<Base64 encoded data>}
```

On the server side, the Base64 encoded data is decoded to raw binary and loaded using OpenCV which then converts it to OpenVINO-friendly data format for inference.
   
Let's see how the `inputs` field in the request body may look like if you decide to send the image:

1) as an array

   ```
   {

   ...

   "inputs": {
      "image": [[[[0.0, 0.0, 128.0] ... [111.0, 102.0, 28.0]]]],
   },

   ...

   }
   ```

2) as binary data

   ```
   {

   ...

   "inputs": {
      "image": { "b64": "aW1hZ2U ... gYnl0ZXM=" },
   },

   ...

   }
   ``` 

## API Reference
- [TensorFlow Serving gRPC API Reference Guide](./model_server_grpc_api_tfs.md)
- [TensorFlow Serving REST API Reference Guide](./model_server_rest_api_tfs.md)

## Usage examples

Examples below assumes OVMS has been started with ResNet50 binary model:

```bash
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -d --rm -v $(pwd)/models:/models -p 8000:8000 -p 9000:9000 openvino/model_server:latest \
--model_name resnet --model_path /models/resnet50 --layout NHWC:NCHW --plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
--port 9000 --rest_port 8000
```

Prepare the client:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/ovmsclient/samples
pip install -r requirements.txt
```

### Run the gRPC client sending the binary input
([see the code](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/ovmsclient/samples/grpc_predict_binary_resnet.py))
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


### Run the REST client sending the binary input
([see the code](https://github.com/openvinotoolkit/model_server/blob/develop/client/python/ovmsclient/samples/http_predict_binary_resnet.py))
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
client, only curl and base64 tool or the requests python package is needed. In case the original input data is jpeg or png 
encoded, there is no preprocessing needed to send the request.

Binary data can significantly reduce the network utilization. In many cases it allows reducing the latency and achieve
very high throughput even with slower network bandwidth.
