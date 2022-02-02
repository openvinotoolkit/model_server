# Support for Binary Input Data {#ovms_docs_binary_input}

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

In case the model was trained with RGB color format and a range other than 0-255, the [Model Optimizer](tf_model_binary_input.md) can apply the required adjustments:
  
`--reverse_input_channels`: Switch the input channels order from RGB to BGR (or vice versa). Applied to original inputs of the model **only** if a number of channels equals 3. Applied after application of --mean_values and --scale_values options, so numbers in --mean_values and  --scale_values go in the order of channels used in the original model  
`--scale` : All input values coming from original network inputs  will be divided by this value. When a list of inputs  is overridden by the --input parameter, this scale is  not applied for any input that does not match with the  original input of the model  
`--mean_values` :  Mean values to be used for the input image per  channel. Values to be provided in the (R,G,B) or (B,G,R) format. Can be defined for desired input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained.

In case of using DAG Scheduler, binary input must be connected to at least one `DL model` node.

Blob data precision from binary input decoding is set automatically based on the target model or the [DAG pipeline](dag_scheduler.md) node.

## API specification

- [gRPC API Reference Guide](./model_server_grpc_api.md)
- [REST API Reference Guide](./model_server_rest_api.md)

## Usage example with binary input

Examples below assumes OVMS has been started with ResNet50 binary model:

```bash
docker run -d --rm -p 8000:8000 -p 9000:9000 openvino/model_server:latest \
--model_name resnet --model_path gs://ovms-public-eu/resnet50-binary --layout NHWC --batch_size 2 --plugin_config '{"CPU_THROUGHPUT_STREAMS": "1"}' \
--port 9000 --rest_port 8000
```

Prepare the client:
```bash
cd model_server/example_client/
pip install -r client_requirements.txt
```

Run the gRPC client sending the binary input:
```
python grpc_binary_client.py --grpc_address localhost --model_name resnet --input_name 0 --output_name 1463 --grpc_port 9000 --images input_images.txt  --batchsize 2
Start processing:
	Model name: resnet
	Images list file: input_images.txt
Batch: 0; Processing time: 22.04 ms; speed 45.38 fps
	 1 airliner 404 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
	 2 white wolf, Arctic wolf, Canis lupus tundrarum 270 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
Batch: 1; Processing time: 15.58 ms; speed 64.16 fps
	 3 bee 309 ; Correct match.
	 4 golden retriever 207 ; Correct match.
Batch: 2; Processing time: 17.93 ms; speed 55.79 fps
	 5 gorilla, Gorilla gorilla 366 ; Correct match.
	 6 magnetic compass 635 ; Correct match.
Batch: 3; Processing time: 17.14 ms; speed 58.36 fps
	 7 peacock 84 ; Correct match.
	 8 pelican 144 ; Correct match.
Batch: 4; Processing time: 15.56 ms; speed 64.25 fps
	 9 snail 113 ; Correct match.
	 10 zebra 340 ; Correct match.
Overall accuracy= 80.0 %
Average latency= 17.2 ms
```

Run the REST API client sending the binary input:
```
python rest_binary_client.py --rest_url http://localhost:8000 --model_name resnet --input_name 0 --output_name 1463  --images input_images.txt  --batchsize 2
Start processing:
	Model name: resnet
	Images list file: input_images.txt
Batch: 0; Processing time: 17.73 ms; speed 56.42 fps
output shape: (2, 1000)
	 1 airliner 404 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
	 2 white wolf, Arctic wolf, Canis lupus tundrarum 270 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
Batch: 1; Processing time: 14.06 ms; speed 71.11 fps
output shape: (2, 1000)
	 3 bee 309 ; Correct match.
	 4 golden retriever 207 ; Correct match.
Batch: 2; Processing time: 14.78 ms; speed 67.66 fps
output shape: (2, 1000)
	 5 gorilla, Gorilla gorilla 366 ; Correct match.
	 6 magnetic compass 635 ; Correct match.
Batch: 3; Processing time: 20.56 ms; speed 48.64 fps
output shape: (2, 1000)
	 7 peacock 84 ; Correct match.
	 8 pelican 144 ; Correct match.
Batch: 4; Processing time: 23.04 ms; speed 43.41 fps
output shape: (2, 1000)
	 9 snail 113 ; Correct match.
	 10 zebra 340 ; Correct match.
Overall accuracy= 80.0 %
Average latency= 17.6 ms
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
