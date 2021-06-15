# Support for binary inputs and outputs data in OpenVINO Model Server

While OpenVINO models don't have the ability to process binary inputs, the model server can accept them and convert
automatically using OpenCV library.

OpenVINO Model Server API allows for sending the request data in a variety of formats inside the TensorProto objects.
Array data is passed inside the tensor_content field, which represent the input data buffer.

When the data is sent in the `string_val` field, it is interpreted as a binary format of the input data.

Note, that while the model metadata reports the inputs shape with layout NHWC, the binary data must be sent with 
shape: [N] with dtype: DT_STRING. Where N represents elements with binary data converted to string bytes.

## Preparing for deployment
Before processing in the target AI model, binary image data is encoded by OVMS to a NHWC layout in BGR color format.
It will also be resized to the model or pipeline node resolution.

Processing the binary image requests requires the model or the custom nodes to accept NHWC layout in BGR color 
format with data with the data range from 0-255. Original layout of the input data can be changed in the 
OVMS configuration in runtime. For example when the orignal model has input shape [1,3,224,224] add a parameter
in the OVMS configuration "layout": "NHWC" or the command line parameter `--layout NHWC`. In result, the model will
have effective shape [1,224,224,3].

In case the model was trained with color format RGB and range other then 0-255, the 
[model optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) 
can apply required adjustments:
--reverse_input_channels: Switch the input channels order from RGB to BGR (or vice versa). Applied to original inputs of the model if and only if a number of channels equals 3. Applied after application of --mean_values and --scale_values options, so numbers in --mean_values and  --scale_values go in the order of channels used in the original model
--scale : All input values coming from original network inputs  will be divided by this value. When a list of inputs  is overridden by the --input parameter, this scale is  not applied for any input that does not match with the  original input of the model
--mean_values :  Mean values to be used for the input image per  channel. Values to be provided in the (R,G,B) or (B,G,R) format. Can be defined for desired input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained.

Alternatively layout conversion can be implemented using a custom node [image transormation](../src/custom_nodes/image_transformation). Custom node can accept
the data in layout NHWC and convert it to NCHW before passing to the target AI model. It can also replace color channel
and rescale the data.

Blob data precision from binary input decoding is set automatically based on the target model or the [DAG pipeline](dag_scheduler.md) node.

## API specification

- [gRPC API Reference Guide](./model_server_grpc_api.md)
- [REST API Reference Guide](./model_server_rest_api.md)

## Usage example with binary input

Examples below assumes OVMS has been started with ResNet50 binary model:

```bash
docker run -d --rm -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path gs://ovms-public-eu/resnet50 --layout NHWC --batch_size 2 --port 9000 --rest_port 8000
```

Run the gRPC client sending the binary input:
```
python grpc_binary_client.py --grpc_address localhost --model_name resnet --input_name 0 --output_name 1463 --grpc_port 9000 --images input_images.txt --iterations 1 --batch_size 2
output results, accuracy and performance
```

Run the REST API client sending the binary input:
```
python rest_binary_client.py --url http://localhost:8000 --model_name resnet --input_name 0 --output_name 1463  --images input_images.txt --iterations 1 --batch_size 2

output results, accuracy and performance
```

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



