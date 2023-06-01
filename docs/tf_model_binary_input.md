# Convert TensorFlow Models to Accept Binary Inputs {#ovms_docs_demo_tensorflow_conversion}

This guide shows how to convert TensorFlow models and deploy them with the OpenVINO Model Server. It also explains how to scale the input tensors and adjust to use binary JPEG or PNG input data.

- In this example TensorFlow model [ResNet](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet) will be used.

- TensorFlow model can be converted into Intermediate Representation format using model_optimizer tool. There are several formats for storing TensorFlow model. In this guide, we present conversion from SavedModel format. More information about conversion process can be found in the [model optimizer guide](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_tutorials.html).

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
- `--mean_values` , `--scale` - should be provided if input pre-processing operations are not a part of topology- and the pre-processing relies on the application providing input data. They can be determined in several ways described in [conversion parameters guide](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html). In this example [model pre-processing script](https://github.com/tensorflow/models/blob/v2.2.0/official/r1/resnet/imagenet_preprocessing.py) was used to determine them.


*Note:* You can find out more about [TensorFlow Model conversion into Intermediate Representation](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) if your model is stored in other formats.

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
cd model_server/client/python/ovmsclient/samples
virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt

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
