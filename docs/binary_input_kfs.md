# Predict on Binary Inputs via KServe API {#ovms_docs_binary_input_kfs}

## GRPC

KServe API allows sending the model input data in a variety of formats inside the [InferTensorContents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#tensor-data-1) objects or in `raw_input_contents` field of [ModelInferRequest](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).
   
When the data is sent in the `bytes_contents` field of `InferTensorContents` and input `datatype` is set to `BYTES`, such input is interpreted as a binary encoded image. The `BYTES` datatype is dedicated to binary encoded **images** and if it's set, the data **must** be placed in `bytes_contents`. Input placed in any other field, including `raw_input_contents` will be ignored, if the datatype is defined as `BYTES`. 

Note, that while the model metadata reports the inputs shape with layout `NHWC`, the binary data must be sent with 
shape: `[N]` with datatype: `BYTES`. Where `N` represents number of images converted to string bytes.

Let's see how the ModelInferRequest object may look like if you decide to send the image:

1) as an array

   ```
   ModelInferRequest {
      model_name: "my_model"
      inputs: [
         {
         datatype: "FP32"
         shape: [3, 300, 300, 3]
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
         shape: [3]
         contents:
            bytes_contents: [[\x31\x92\ ... \xaa\x4a], [\x00\x00\ ... \xff\xff]]
         }
      ]
   }
   ```

When sending data in the array format, the shape and datatype gives information on how to interpret bytes in the contents. For binary encoded data, the only information given by the `shape` field is the amount of images in the batch. On the server side, the bytes in each element of the `bytes_contents` field are loaded, resized to match model input shape and converted to the OpenVINO-friendly array format by OpenCV.

## HTTP

### JPEG / PNG encoded images

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
The list must be formed as a string, so for example, for 3 images in the batch, you may pass - `"9821,12302,7889"`.
If the request contains only one input `binary_data_size` parameter can be omitted - in this case whole buffer is treated as a input image.

For HTTP request headers, `Inference-Header-Content-Length` header must be provided to give the length of the JSON object, and `Content-Length` continues to give the full body length (as HTTP requires). See an extended example with the request headers, and multiple images in the batch:

```
POST /v2/models/my_model/infer HTTP/1.1
Host: localhost:5000
```
```JSON
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
<12302 bytes of the second image in the batch for model_input tensor>
<7889 bytes of the third image in the batch for model_input tensor>
```

On the server side, the binary encoded data is loaded using OpenCV which then converts it to OpenVINO-friendly data format for inference.

The structure of the response is specified [Inference Response specification](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-response-json-object).

### Raw data

Above section described how to send JPEG/PNG encoded image via REST interface. Data sent like this is processed by OpenCV to convert it to OpenVINO-friendly format. Many times data is already available in OpenVINO-friendly format and all you want to do is to send it and get the prediction.

With KServe API you can also send raw data in a binary representation via REST interface. **That way the request gets smaller and easier to process on the server side, therefore using this format is more effecient when working with RESTful API, than providing the input data in a JSON object**. To send raw data in the binary format, you need to specify `datatype` other than `BYTES` and data `shape`, should match the input `shape` (also the memory layout should be compatible). 

Getting back to the example from the previous section with 3 images in a batch, let's assume they are not JPEGs or PNGs, but raw array with layout NHWC. The request with such data could look like this:

```
POST /v2/models/my_model/infer HTTP/1.1
Host: localhost:5000
```
```JSON
Content-Type: application/octet-stream
Inference-Header-Content-Length: <xx>
Content-Length: <xx+(3 x 1080000)>
{
"model_name" : "my_model",
"inputs" : [
   {
      "name" : "model_input",
      "shape" : [ 3, 300, 300, 3 ],
      "datatype" : "FP32",
      "parameters" : {
         "binary_data_size" : "3240000"
      }
   },

]
}
<3240000 bytes of the whole data batch for model_input tensor>
```

For the Raw Data binary inputs `binary_data_size` parameter can be omitted since the size of particular input can be calculated from its shape.

### Binary Outputs

Outputs of response can be send in binary format similar to the binary inputs. To force a output to be sent in binary format you need to use "binary_data" : true parameter in request JSON. For example:
```JSON
{
  "model_name" : "mymodel",
  "inputs" : [...],
  "outputs" : [
    {
      "name" : "output0",
      "parameters" : {
        "binary_data" : true
      }
    }
  ]
}
```

Assuming the output datatype is FP32 and shape is [ 2, 2 ] response to this request would be:

```JSON
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Inference-Header-Content-Length: <yy>
Content-Length: <yy+16>
{
  "outputs" : [
    {
      "name" : "output0",
      "shape" : [ 2, 2 ],
      "datatype"  : "FP32",
      "parameters" : {
        "binary_data_size" : 16
      }
    }
  ]
}
<16 bytes of data for output0 tensor>
```


## API Reference

- [Kserve gRPC API Reference Guide](./model_server_grpc_api_kfs.md)
- [Kserve REST API Reference Guide](./model_server_rest_api_kfs.md)

## Usage examples

Examples below assumes OVMS has been started with ResNet50 binary model:

```bash
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -d -u $(id -u) -v $(pwd)/models:/models -p 8000:8000 -p 9000:9000 openvino/model_server:latest \
--model_name resnet --model_path /models/resnet50 --layout NHWC:NCHW --plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
--port 9000 --rest_port 8000
```

Prepare the client:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/kserve-api/samples
pip install -r requirements.txt
```

### Run the gRPC client sending JPEG images
([see the code](https://github.com/openvinotoolkit/model_server/blob/v2022.3/client/python/kserve-api/samples/grpc_infer_binary_resnet.py))
```bash
python3 ./grpc_infer_binary_resnet.py --grpc_port 9000 --images_list resnet_input_images.txt --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet
Start processing:
        Model name: resnet
Iteration 0; Processing time: 13.36 ms; speed 74.82 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 1; Processing time: 14.51 ms; speed 68.92 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 2; Processing time: 10.14 ms; speed 98.60 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 3; Processing time: 9.06 ms; speed 110.31 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 4; Processing time: 8.44 ms; speed 118.51 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 5; Processing time: 19.27 ms; speed 51.89 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 6; Processing time: 11.48 ms; speed 87.12 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 7; Processing time: 10.64 ms; speed 94.03 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 8; Processing time: 11.89 ms; speed 84.10 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 9; Processing time: 11.35 ms; speed 88.11 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 11.60 ms; average speed: 86.21 fps
median time: 11.00 ms; median speed: 90.91 fps
max time: 19.00 ms; min speed: 52.63 fps
min time: 8.00 ms; max speed: 125.00 fps
time percentile 90: 14.50 ms; speed percentile 90: 68.97 fps
time percentile 50: 11.00 ms; speed percentile 50: 90.91 fps
time standard deviation: 2.97
time variance: 8.84
Classification accuracy: 100.00
```


### Run the REST client sending JPEG images
([see the code](https://github.com/openvinotoolkit/model_server/blob/v2022.3/client/python/kserve-api/samples/http_infer_binary_resnet.py))
```bash
python3 ./http_infer_binary_resnet.py --http_port 8000 --images_list resnet_input_images.txt --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet
Start processing:
        Model name: resnet
Iteration 0; Processing time: 16.70 ms; speed 59.89 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 1; Processing time: 16.03 ms; speed 62.39 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 2; Processing time: 14.23 ms; speed 70.29 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 3; Processing time: 12.33 ms; speed 81.11 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 4; Processing time: 11.59 ms; speed 86.30 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 5; Processing time: 11.67 ms; speed 85.69 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 6; Processing time: 12.51 ms; speed 79.92 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 7; Processing time: 10.98 ms; speed 91.07 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 8; Processing time: 10.59 ms; speed 94.44 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 9; Processing time: 14.45 ms; speed 69.22 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 12.60 ms; average speed: 79.37 fps
median time: 12.00 ms; median speed: 83.33 fps
max time: 16.00 ms; min speed: 62.50 fps
min time: 10.00 ms; max speed: 100.00 fps
time percentile 90: 16.00 ms; speed percentile 90: 62.50 fps
time percentile 50: 12.00 ms; speed percentile 50: 83.33 fps
time standard deviation: 2.15
time variance: 4.64
Classification accuracy: 100.00
```

### Run the REST client with raw data sent in binary representation
([see the code](https://github.com/openvinotoolkit/model_server/blob/v2022.3/client/python/kserve-api/samples/http_infer_resnet.py))
```bash
python3 ./http_infer_resnet.py --http_port 8000 --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False --binary_data
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs_nhwc.npy
        Numpy file shape: (10, 224, 224, 3)

Iteration 1; Processing time: 36.58 ms; speed 27.34 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 33.76 ms; speed 29.62 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 28.55 ms; speed 35.03 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 28.27 ms; speed 35.37 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 28.83 ms; speed 34.69 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 26.80 ms; speed 37.31 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 27.20 ms; speed 36.76 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 26.46 ms; speed 37.80 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 29.52 ms; speed 33.87 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 27.49 ms; speed 36.37 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 28.80 ms; average speed: 34.72 fps
median time: 28.00 ms; median speed: 35.71 fps
max time: 36.00 ms; min speed: 27.78 fps
min time: 26.00 ms; max speed: 38.46 fps
time percentile 90: 33.30 ms; speed percentile 90: 30.03 fps
time percentile 50: 28.00 ms; speed percentile 50: 35.71 fps
time standard deviation: 3.06
time variance: 9.36
Classification accuracy: 100.00
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
