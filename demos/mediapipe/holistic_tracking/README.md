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

git clone https://github.com/openvinotoolkit/mediapipe.git
git co atobisze_hollistic
cd mediapipe
make docker_build
cd ..

docker run -it -v `pwd`:/models mediapipe_ovms:latest bash
python setup_ovms.py --get_models --get_graphs --convert_pose
cp -rf mediapipe/models/ovms /models
exit

sudo chown $(id -u) ovms
cp config_holistic.json ovms
cp holistic_tracking.pbtxt ovms
```

## Run OpenVINO Model Server
```bash
docker run -it -v $PWD/ovms:/models -p 9022:9022 openvino/model_server:latest --config_path /models/config_holistic.json --port 9022
```

## Run client application
```bash
python mediapipe_holistic_tracking.py
```