# TF model conversion to IR with scaling for binary input

This document guides how to prepare TensorFlow model to use binary input with OpenVINO&trade; Model Server.

- In this example TF model that will be converted is [ResNet in TensorFlow](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet).

- To convert TensorFlow model into Intermediate Representation format model_optimizer tool can be used. There are several ways to store TensorFlow model and in this guide we are going to convert SavedModel format. More information about conversion process can be found on the openVINO&trade; documentation in [Converting a TensorFlow* Model](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#savedmodel_format) guide.

- Sending requests with binary input requires some adjustments in deployment for both OVMS and model. More informations can be found on [Support for binary inputs data in OpenVINO Model Server](https://github.com/openvinotoolkit/model_server/blob/main/docs/binary_input.md).

## Steps

### Prepering the Model

Download the model
```Bash
wget http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz

tar -xvzf resnet_v2_fp32_savedmodel_NHWC.tar.gz 

mv resnet_v2_fp32_savedmodel_NHWC/1538687283/ resnet_v2
rmdir resnet_v2_fp32_savedmodel_NHWC/
```
*Note:* Directories operations are not neccessery for the prepartion, but in this guide in order to keep the commands as simple as possible the directories are simplified.

Pull the latest openvino ubuntu_dev image from dockerhub
```Bash
docker pull openvino/ubuntu18_dev:latest
```

Convert the TensorFlow model to Intermediate Representation format using model_optimizer tool:
```Bash
docker run -u $(id -u):$(id -g) -v `pwd`/resnet_v2/:/resnet openvino/ubuntu18_dev:latest deployment_tools/model_optimizer/mo.py --saved_model_dir /resnet/ --output_dir /resnet/IR/1/ --input_shape=[1,224,224,3] --mean_values=[123.68,116.78,103.94] --reverse_input_channels
```

*Note:* You can find out more about [TensorFlow Model conversion into Intermediate Representation](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) if your model is stored in other formats.

This operation will create model files in `${PWD}/resnet_v2/IR/1/` folder.
```Bash
saved_model.bin
saved_model.mapping
saved_model.xml
```

Create `models/resnet/1/` directory and paste `.bin` and `.xml` files into it

### OVMS Deployment

Pull the latest openVINO&trade; Model Server image
```Bash
docker pull openvino/model_server:latest
```

Deploy OVMS using the following command:
```Bash
docker run -p 9000:9000 -v ${PWD}/models:/models openvino/model_server:latest --model_path /models/resnet --model_name resnet --port 9000 --layout NHWC
```

### Requesting the Service

```Bash
cd example_client
virtualenv .venv
. .venv/bin/activate
pip install -r client_requirements.txt

python grpc_binary_client.py --images_list input_images.txt --grpc_port 9000 --input_name input_tensor --output_name  softmax_tensor --model_name resnet
```

### Output of the Script

```Bash
Start processing:
        Model name: resnet
        Images list file: input_images.txt
Batch: 0; Processing time: 107.76 ms; speed 9.28 fps
         1 airliner 404 ; Correct match.
Batch: 1; Processing time: 101.90 ms; speed 9.81 fps
         2 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Batch: 2; Processing time: 100.58 ms; speed 9.94 fps
         3 bee 309 ; Correct match.
Batch: 3; Processing time: 101.75 ms; speed 9.83 fps
         4 golden retriever 207 ; Correct match.
Batch: 4; Processing time: 94.31 ms; speed 10.60 fps
         5 gorilla, Gorilla gorilla 366 ; Correct match.
Batch: 5; Processing time: 96.41 ms; speed 10.37 fps
         6 magnetic compass 635 ; Correct match.
Batch: 6; Processing time: 93.74 ms; speed 10.67 fps
         7 peacock 84 ; Correct match.
Batch: 7; Processing time: 94.35 ms; speed 10.60 fps
         8 pelican 144 ; Correct match.
Batch: 8; Processing time: 95.32 ms; speed 10.49 fps
         9 snail 113 ; Correct match.
Batch: 9; Processing time: 101.97 ms; speed 9.81 fps
         10 zebra 340 ; Correct match.
Overall accuracy= 100.0 %
Average latency= 98.2 ms
```