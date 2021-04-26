# Support for binary inputs and outputs data in OpenVINO Model Server

While OpenVINO models don't have ability to process binary inputs, the model server can accept them and convert
automatically using OpenCV library.

OpenVINO Model Server API allows for sending the request data in a variety of formats inside the TensorProto objects.
Array data are passed inside the tensor_contnet field, which represent the input data buffer.

When the data is sent in the field `string_val`, it is interpreted as a binary format of the input data.

Note that the model metadata reports the inputs shape with layout NHWC, the binary data must be sent with 
shape: [N] with dtype: DT_STRING. N represents elements with binary data converted to string bytes.

## Preparing for deployment
Before processing in the target model, it will be read by OpenCV and passed are a blob. The data converted by OpenCV
has always layout NHWC. It will be also resize to the target model of pipeline node resolution.

The model processing binary data needs to be set in NHWC layout. Original layout of the input data can be changed in the 
OVMS configuration in runtime. 
Alternatively layout conversion can be implemented using a custom on `image transormation`.

Note that OpenCV generates the blob data with the normalization 0-255 and BGR color format. In case the model was 
trained with different dataset, execution could include `image transormation` custom node to adjust the data to the target model.

Blob data precision from binary input decoding is set automatically based on the target model or the DAG pipeline node.

## Returning binary outputs

Some models or DAG pipelines return images in the response. OpenVINO model outputs are always in the form for arrays. It is possible,
however, to configure OVMS to send the image outputs in the binary format instead.

The binary data will be created out of OpenVINO response blob using OpenCV library. Images in the response can be returned 
as JPEG encoded. REST API responses will be in addition to that also Base64 encoded.

Binary outputs can be enabled by setting the output name with `_binary` suffix. In case the model is already exported with
different output name, OVMS has option to configure inputs and outputs names mapping by creating a json file `mapping_config.json`
and planning it together with the model files in the model version folder.
```json
{
  "inputs": { "tensor_input":"tensor_input" }, 
  "outputs": {"tensor_name":"tensor_name_binary" }
}
```
Binary outputs can be successfully returned only when they include images with layout NHWC, color format BGR and data range
0-255. If the model has different format of the results, it can be postprocess by a custom node like `image_transformation`
with the DAG pipeline.

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
## Usage example with binary input and output

download superresolution model https://docs.openvinotoolkit.org/latest/omz_models_model_text_image_super_resolution_0001.html
add tensor_mapping.json with binary_sufix
Start model
Start client
report super resolution results in jpeg

## Error handling:
In case the binary input can not be converted to the array of correct shape, an error status is returned:
- 400   BAD_REQUEST for REST API
- 3 INVALID_ARGUMENT for gRPC API

When the model outputs with `_binary` suffix can not be JPEG encoded, the following errors will be sent:
- 412 Precondition Failed for REST API
- 9 FAILED_PRECONDITION for gRPC API

## Recommendations:

Sending the data in binary format can significantly simplify the client code and it's preprocessing load. With the REST API
client, there is just needed curl and base64 tool or requests python package. In case the original input data is jpeg or png 
encoded, there is no preprocessing needed to send the request.

Binary data can significantly reduce the network utilization. In many case it allows reducing the latency and achieve
very higher throughout even with slower network bandwidth.



