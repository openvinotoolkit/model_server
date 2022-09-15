# Support for Binary Input Data {#ovms_docs_binary_input}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_demo_tensorflow_conversion

@endsphinxdirective

While OpenVINO models don't have the ability to process images directly in their binary format, the model server can accept them and convert
automatically from JPEG/PNG to OpenVINO friendly format using OpenCV library.

## Preparing for deployment
Before processing in the target AI model, binary image data is encoded by OVMS to a NHWC layout in BGR color format.
It will also be resized to the model or pipeline node resolution. When the model resolution supports range of values and image data shape is out of range it will be adjusted to the nearer border. For example, when model shape is: [1,100:200,200,3]:

- if input shape is [1,90,200,3] it will be resized into [1,100,200,3]
- if input shape is [1,220,200,3] it will be resized into [1,200,200,3]

In order to use binary input functionality, model or pipeline input layout needs to be compatible with `N...HWC` and have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions. It means that input layout needs to resemble `NHWC` layout, e.g. default `N...` will work. On the other hand, binary image input is not supported for inputs with `NCHW` layout. 

To fully utilize binary input utility, automatic image size alignment will be done by OVMS when:
- input shape does not include dynamic dimension value (`-1`)
- input layout is configured to be either `...` (custom nodes) and `NHWC` or `N?HWC` (or `N?HWC`, when modified by a [demultiplexer](demultiplexing.md))

Processing the binary image requests requires the model or the custom nodes to accept BGR color 
format with data with the data range from 0-255. Original layout of the input data can be changed in the 
OVMS configuration in runtime. For example when the orignal model has input shape [1,3,224,224] add a parameter
in the OVMS configuration "layout": "NHWC:NCHW" or the command line parameter `--layout NHWC:NCHW`. In result, the model will
have effective shape [1,224,224,3] and layout `NHWC`.

In case the model was trained with RGB color format and a range other than 0-255, the [Model Optimizer](tf_model_binary_input.md) can apply the required adjustments:
  
`--reverse_input_channels`: Switch the input channels order from RGB to BGR (or vice versa). Applied to original inputs of the model **only** if a number of channels equals 3. Applied after application of --mean_values and --scale_values options, so numbers in --mean_values and  --scale_values go in the order of channels used in the original model  
`--scale` : All input values coming from original network inputs  will be divided by this value. When a list of inputs  is overridden by the --input parameter, this scale is  not applied for any input that does not match with the  original input of the model  
`--mean_values` :  Mean values to be used for the input image per  channel. Values to be provided in the (R,G,B) or (B,G,R) format. Can be defined for desired input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained.

Blob data precision from binary input decoding is set automatically based on the target model or the [DAG pipeline](dag_scheduler.md) node.

## Request prediction on binary encoded image

OpenVINO Model Server exposes TensorFlow Serving as well as KServe compatible APIs. There is a difference in how those APIs handle binary input data, so please select the API you are working with in the tab below to learn more.

@sphinxdirective

.. tab:: TensorFlow Serving API 

   ### GRPC

   TensorFlow Serving API allows for sending the request data in a variety of formats inside the [TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) objects.
   Array data is passed inside the `tensor_content` field, which represents the input data buffer.

   When the data is sent in the `string_val` field, it is interpreted as a binary format of the input data.

   Note, that while the model metadata reports the inputs shape with layout NHWC, the binary data must be sent with 
   shape: [N] with dtype: DT_STRING. Where N represents number of images converted to string bytes.

   Let's see how the TensorProto object may look like if you decide to send the image:

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

   ### HTTP

   TensorFlow Serving API also allows sending binary encoded data via HTTP interface. The binary data needs to be Base64 encoded and put into `inputs` or `instances` field as a map in form:

   ```<input_name>: {"b64":<Base64 encoded data>}```

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
      

.. tab:: KServe API

   ### GRPC

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

   ### HTTP

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

@endsphinxdirective


- [TensorFlow Serving gRPC API Reference Guide](./model_server_grpc_api_tfs.md)
- [Kserve gRPC API Reference Guide](./model_server_grpc_api_kfs.md)
- [TensorFlow Serving REST API Reference Guide](./model_server_rest_api_tfs.md)
- [Kserve REST API Reference Guide](./model_server_rest_api_kfs.md)

## Usage example with binary input

Examples below assumes OVMS has been started with ResNet50 binary model:

```bash
docker run -d --rm -p 8000:8000 -p 9000:9000 openvino/model_server:latest \
--model_name resnet --model_path gs://ovms-public-eu/resnet50-binary --layout NHWC:NCHW --plugin_config '{"CPU_THROUGHPUT_STREAMS": "1"}' \
--port 9000 --rest_port 8000
```
@sphinxdirective

.. tab:: TensorFlow Serving API  

Prepare the client:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/ovmsclient/samples
pip install -r requirements.txt
```

Run the gRPC client sending the binary input:
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


Run the REST client sending the binary input:
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

.. tab:: KServe API

TODO

@endsphinxdirective

## Error handling:
In case the binary input can not be converted to the array of correct shape, an error status is returned:
- 400 - BAD_REQUEST for REST API
- 3 - INVALID_ARGUMENT for gRPC API


## Recommendations:

Sending the data in binary format can significantly simplify the client code and it's preprocessing load. With the REST API
client, curl and base64 tool or the requests python package is just needed. In case of the original input data is jpeg or png 
encoded, there is no preprocessing needed to send the request.

Binary data can significantly reduce the network utilization. In many cases it allows reducing the latency and achieve
very high throughput even with slower network bandwidth.
