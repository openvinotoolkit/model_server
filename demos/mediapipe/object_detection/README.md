# MediaPipe Object Detection Demo {#ovms_docs_demo_mediapipe_object_detection}

This guide shows how to implement [MediaPipe](../../../docs/mediapipe.md) graph using OVMS.

Example usage of graph that accepts Mediapipe::ImageFrame as a input:


## Prepare the repository

Clone the repository and enter mediapipe object_detection directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/mediapipe/object_detection
```

## Prepare models and the environment:
```bash
virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt

python mediapipe_object_detection.py --download_models
```

## Run OpenVINO Model Server
```bash
docker run -d -v $PWD/ovms:/demo -p 9000:9000 openvino/model_server:latest --config_path /demo/config.json --port 9000
```

## Run the client:
```bash
python mediapipe_object_detection.py --grpc_port 9000 --images ./input_images.txt
Start processing:
        Graph name: objectDetection
airliner.jpeg
Iteration 0; Processing time: 26.75 ms; speed 37.39 fps
arctic-fox.jpeg
Iteration 1; Processing time: 24.87 ms; speed 40.21 fps
bee.jpeg
Iteration 2; Processing time: 21.79 ms; speed 45.89 fps
golden_retriever.jpeg
Iteration 3; Processing time: 22.70 ms; speed 44.05 fps
gorilla.jpeg
Iteration 4; Processing time: 22.25 ms; speed 44.95 fps
magnetic_compass.jpeg
Iteration 5; Processing time: 22.29 ms; speed 44.86 fps
peacock.jpeg
Iteration 6; Processing time: 22.99 ms; speed 43.50 fps
pelican.jpeg
Iteration 7; Processing time: 22.70 ms; speed 44.05 fps
snail.jpeg
Iteration 8; Processing time: 23.35 ms; speed 42.82 fps
zebra.jpeg
Iteration 9; Processing time: 21.97 ms; speed 45.51 fps
```
Received images with bounding boxes will be located in ./results directory.