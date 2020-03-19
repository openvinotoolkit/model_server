# Getting started with OVMS

## Requirements
* docker version >= 19.03
* git version >= 1.8.3
* python 3.6
* pip3 version >= 20.0.2 with setuptools >= 46.0.0
* wget version >= 1.14

### OpenVino Model Server
**1. Clone repository**  

Assumption: Access to github.com based on ssh key (see: https://help.github.com/articles/connecting-to-github-with-ssh/)
```
git clone ssh://git@github.com:IntelAI/OpenVINO-model-server.git
```
**2. OVMS image preparation**  

Download OVMS image using docker (from: https://hub.docker.com/r/intelaipg/openvino-model-server):
```
docker pull intelaipg/openvino-model-server
```
or build image via [make](https://github.com/IntelAI/OpenVINO-model-server#building):
```
make docker_build_apt_ubuntu
```

**3. Model preparation - resnet example**  

Correct model directory structure:
```bash
models/
├── model/
    ├── 1/
        ├── ir_model.bin
        └── ir_model.xml
```
Prepare directory according to the structure above:
```
mkdir -p models/model/1/
```
Move to the created direcory:
```
cd models/model/1/
```
Download model files (.xml & .bin) - resnet example
```
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
```
```
wget https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
```
**4. Run OpenVino Model Server**  

Start OVMS using docker - [details](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.md):
```
docker run --rm -d -v /path/to/models:/docker/models/path -p <port>:<port> <rest_port>:<rest_port> ie-serving-py \
/ie-serving-py/start_server.sh ie_serving model --model_path /docker/models/path --model_name <model_name> --port <port> --rest_port <rest_port>
```
**Returns: <container_id>**

Stop OVMS using docker:
```
docker stop <container_id>
```

### OpenVino Model Server - client

Move to main repository directory - OpenVINO-model-server.

**1. Set up virtualenv**  

Create venv:
```
python3 -m venv <venv_name>
```
Activate created venv:
```
source <venv_name>/bin/activate
```
Install requirements:
```
pip3 install -r requirements.txt
```
**2. Run OVMS client - [details](https://github.com/IntelAI/OpenVINO-model-server/tree/master/example_client)**
```
python3 example_client/grpc_serving_client.py --grpc_address <grpc_address> --grpc_port <grpc_port> --images_numpy_path imgs.npy --input_name 0 --output_name 1463 --transpose_input False --labels_numpy lbs.npy --model_name resnet50-binary-0001
```
