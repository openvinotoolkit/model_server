# MediaPipe Object Detection Demo {#ovms_docs_demo_mediapipe_object_detection}

This guide shows how to implement [MediaPipe](../../../docs/mediapipe.md) graph using OVMS.

Example usage of graph that accepts Mediapipe::ImageFrame as a input:

## Prepare the repository

Clone the repository and enter mediapipe object_detection directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/mediapipe/holistic_tracking

virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt

curl -kL -o girl.jpg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg

curl -o download_utils.py -kL https://github.com/openvinotoolkit/mediapipe/blob/main/mediapipe/python/solutions/download_utils.py

docker run -it -v `pwd`:/models mediapipe_ovms:latest bash
python setup_ovms.py --get_models --convert_pose
cp -rf mediapipe/models/ovms /models
mkdir -p /models/mediapipe/modules/hand_landmark/
cp -rf mediapipe/modules/hand_landmark/handedness.txt /models/mediapipe/modules/hand_landmark/

sudo chown -R $(id -u) ovms
cp config_holistic.json ovms
cp holistic_tracking.pbtxt ovms
```

### Pull the Latest Model Server Image
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub :
```Bash
docker pull openvino/model_server:latest
```

## Run OpenVINO Model Server
```bash
docker run -it -v $PWD/ovms:/models -p 9022:9022 openvino/model_server:latest --config_path /models/config_holistic.json --port 9022
```

## Run client application
```bash
python mediapipe_holistic_tracking.py
```