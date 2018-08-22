# gRPC client example

This client can be used to demonstrate connectivity with ie-serving service over gRPC API and
 TensorFlow server using Predict call.


## Requirements

Install client pip dependencies depending on the python version  

## Usage:

```
python grpc_serving_client.py --help
usage: grpc_serving_client.py [-h] --images_numpy_path IMAGES_NUMPY_PATH
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--transpose_input {False,True}] [--iterations]
                              [--model_name]

Do requests to ie_serving and tf_serving using images in numpy format

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c]
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service
  --grpc_port GRPC_PORT
                        Specify port to grpc service
  --input_name INPUT_NAME
                        Specify input tensor name
  --output_name OUTPUT_NAME
                        Specify output name
  --transpose_input {False,True}
                        Set to False to skip NHWC->NCHW input transposing
  --iterations          Number of requests iterations, as default use number
                        of images in numpy memmap.
  --model_name          Define model name, must be same as is in service                       
(.python_serving) ~/ie-serving-py/example_client$

```

## Examples:

```
python grpc_serving_client.py --images_numpy_path imgs.npy --grpc_address localhost --grpc_port 9000

python grpc_serving_client.py --images_numpy_path imgs.npy --grpc_address localhost --grpc_port 9000 --transpose_input False

python grpc_serving_client.py --images_numpy_path imgs.npy --grpc_address localhost --grpc_port 9000 \
--input_name input --output_name resnet_v1_152/predictions/Reshape_1

```