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
Iteration 0; Processing time: 41.05 ms; speed 24.36 fps
golden_retriever.jpeg
Iteration 1; Processing time: 25.04 ms; speed 39.93 fps
pelican.jpeg
Iteration 2; Processing time: 29.88 ms; speed 33.46 fps
zebra.jpeg
Iteration 3; Processing time: 26.61 ms; speed 37.59 fps
```
Received images with bounding boxes will be located in ./results directory.

## Real time stream analysis

For demo featuring real time stream application see [real_time_stream_analysis](https://github.com/openvinotoolkit/model_server/tree/main/demos/real_time_stream_analysis/python)
