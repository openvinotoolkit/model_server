# Convert TensorFlow Models to Accept Binary Inputs {#ovms_docs_demo_tensorflow_conversion}

This guide shows how to convert TensorFlow models and deploy them with the OpenVINO Model Server. It also explains how to scale the input tensors and adjust to use binary JPEG or PNG input data.

- In this example TensorFlow model [ResNet](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet) will be used.

- TensorFlow model can be converted into Intermediate Representation format using model_optimizer tool. There are several formats for storing TensorFlow model. In this guide, we present conversion from SavedModel format. More information about conversion process can be found in the [model optimizer guide](https://docs.openvino.ai/2025/openvino-workflow/model-preparation.html).

- Binary input format has several requirements for the model and ovms configuration. More information can be found in [binary inputs documentation](binary_input.md).
## Steps

### Preparing the Model

Download the model
```bash
wget http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz

tar -xvzf resnet_v2_fp32_savedmodel_NHWC.tar.gz

mv resnet_v2_fp32_savedmodel_NHWC/1538687283/ resnet_v2
rmdir resnet_v2_fp32_savedmodel_NHWC/
```
*Note:* Directories operations are not necessary for the preparation, but in this guide the directories are simplified.

Convert the TensorFlow model to Intermediate Representation format using model_optimizer tool:
```bash
docker run -u $(id -u):$(id -g) -v ${PWD}/resnet_v2/:/resnet openvino/ubuntu20_dev:2022.1.0 mo --saved_model_dir /resnet/ --output_dir /resnet/models/resnet/1/ --input_shape=[1,224,224,3] --mean_values=[123.68,116.78,103.94] --reverse_input_channels
```

*Note:* Some models might require other parameters such as `--scale` parameter.
- `--reverse_input_channels` - required for models that are trained with images in RGB order.
- `--mean_values` , `--scale` - should be provided if input pre-processing operations are not a part of topology- and the pre-processing relies on the application providing input data. They can be determined in several ways described in [conversion parameters guide](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-tensorflow.html). In this example [model pre-processing script](https://github.com/tensorflow/models/blob/v2.2.0/official/r1/resnet/imagenet_preprocessing.py) was used to determine them.


*Note:* You can find out more about [TensorFlow Model conversion into Intermediate Representation](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-tensorflow.html) if your model is stored in other formats.

This operation will create model files in `${PWD}/resnet_v2/models/resnet/1/` folder.
```bash
tree resnet_v2/models/resnet/1
resnet_v2/models/resnet/1
├── saved_model.bin
├── saved_model.mapping
└── saved_model.xml
```

### OVMS Deployment
Pull the latest openvino model_server image from dockerhub
```bash
docker pull openvino/model_server:latest
```

Deploy OVMS using the following command:
```bash
docker run -d -p 9000:9000 -v ${PWD}/resnet_v2/models:/models openvino/model_server:latest --model_path /models/resnet --model_name resnet --port 9000 --layout NHWC
```

**Note:** This model has `N...` layout by default, but binary inputs feature requires model to have `NHWC` or `N?HWC` layout, therefore we specify `--layout NHWC` option.

### Running the inference requests from the client

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd client/python/tensorflow-serving-api/samples
virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt
python grpc_predict_binary_resnet.py --grpc_address localhost --model_name resnet --input_name input_tensor --output_name softmax_tensor:0 --grpc_port 9000 --images ../../resnet_input_images.txt

Start processing:
        Model name: resnet
        Images list file: ../../resnet_input_images.txt
Batch: 0; Processing time: 17.62 ms; speed 56.76 fps
         1 airliner 404 ; Correct match.
Batch: 1; Processing time: 13.20 ms; speed 75.75 fps
         2 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Batch: 2; Processing time: 11.60 ms; speed 86.19 fps
         3 bee 309 ; Correct match.
Batch: 3; Processing time: 10.48 ms; speed 95.37 fps
         4 golden retriever 207 ; Correct match.
Batch: 4; Processing time: 9.73 ms; speed 102.76 fps
         5 gorilla, Gorilla gorilla 366 ; Correct match.
Batch: 5; Processing time: 9.71 ms; speed 103.04 fps
         6 magnetic compass 635 ; Correct match.
Batch: 6; Processing time: 9.84 ms; speed 101.64 fps
         7 peacock 84 ; Correct match.
Batch: 7; Processing time: 9.60 ms; speed 104.16 fps
         8 pelican 144 ; Correct match.
Batch: 8; Processing time: 9.86 ms; speed 101.42 fps
         9 snail 113 ; Correct match.
Batch: 9; Processing time: 10.68 ms; speed 93.68 fps
         10 zebra 340 ; Correct match.
Overall accuracy= 100.0 %
Average latency= 10.6 ms
```