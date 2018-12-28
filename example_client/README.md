# gRPC client examples

This client can be used to demonstrate connectivity with ie-serving service over gRPC API and
 TensorFlow server using Predict call.


## Requirements

Install client pip dependencies depending on the python version  

## Usage:

```bash
python grpc_serving_client.py --help
usage: grpc_serving_client.py [-h] --images_numpy_path IMAGES_NUMPY_PATH
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--transpose_input {False,True}]
                              [--iterations ITERATIONS]
                              [--batchsize BATCHSIZE]
                              [--model_name MODEL_NAME]

Do requests to ie_serving and tf_serving using images in numpy format

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c]
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default:
                        resnet_v1_50/predictions/Reshape_1
  --transpose_input {False,True}
                        Set to False to skip NHWC->NCHW input transposing.
                        default: True
  --iterations ITERATIONS
                        Number of requests iterations, as default use number
                        of images in numpy memmap. default:0 (consume all
                        frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service.
                        default: resnet
```

```bash
python get_serving_meta.py --help
usage: get_serving_meta.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT] [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

Get information about served models

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service.
                        default: resnet
  --model_version MODEL_VERSION
                        Define model version - must be numerical
```
## Examples:

```bash
python grpc_serving_client.py --images_numpy_path imgs.npy --grpc_address localhost --grpc_port 9000

python grpc_serving_client.py --images_numpy_path imgs.npy --grpc_address localhost --grpc_port 9000 --transpose_input False

python grpc_serving_client.py --images_numpy_path imgs.npy --grpc_address localhost --grpc_port 9000 \
--input_name input --output_name resnet_v1_152/predictions/Reshape_1

python get_serving_meta.py --grpc_port 9003 --model_name resnet --model_version 1

```