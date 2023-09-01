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

python mediapipe_holistic_tracking.py --download_models
```

The models setup should look like this
```bash
ovms
├── config_holistic.json
├── face_detection_short_range
│   └── 1
│       └── face_detection_short_range.tflite
├── face_landmark
│   └── 1
│       └── face_landmark.tflite
├── hand_landmark_full
│   └── 1
│       └── hand_landmark_full.tflite
├── hand_recrop
│   └── 1
│       └── hand_recrop.tflite
├── holistic_tracking.pbtxt
├── iris_landmark
│   └── 1
│       └── iris_landmark.tflite
├── palm_detection_full
│   └── 1
│       └── palm_detection_full.tflite
├── pose_detection
│   └── 1
│       └── pose_detection.tflite
└── pose_landmark_full
    └── 1
        └── pose_landmark_full.tflite
```

### Pull the Latest Model Server Image
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub :
```Bash
docker pull openvino/model_server:latest

```

## Run OpenVINO Model Server
```bash
docker run -it -v $PWD/mediapipe:/mediapipe -v $PWD/ovms:/models -p 9000:9000 openvino/model_server:latest --config_path /models/config_holistic.json --port 9000
```

## Run client application for holistic tracking - default demo
```bash
python mediapipe_holistic_tracking.py --grpc_port 9000
```

## Output image
![output](output_image.jpg)

## Run client application for iris tracking
```bash
python mediapipe_holistic_tracking.py --graph_name irisTracking
```

## Output image
![output](output_image1.jpg)
## RTSP Client

Build docker image containing rtsp client along with its dependencies
The rtsp client app needs to have access to RTSP stream to read from and write to.

Example rtsp server [mediamtx](https://github.com/bluenviron/mediamtx)

Then write to the server using ffmpeg

```bash
ffmpeg -f dshow -i video="HP HD Camera" -f rtsp -rtsp_transport tcp rtsp://localhost:8080/channel1
```

```bash
docker build ../../common/stream_client/ -t rtsp_client
```

### Start the client

- Command

```bash
docker run -v $(pwd):/workspace rtsp_client --help
usage: rtsp_client.py [-h] [--grpc_address GRPC_ADDRESS]
                      [--input_stream INPUT_STREAM]
                      [--output_stream OUTPUT_STREAM]
                      [--model_name MODEL_NAME] [--verbose VERBOSE]
                      [--input_name INPUT_NAME]

options:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service
  --input_stream INPUT_STREAM
                        Url of input rtsp stream
  --output_stream OUTPUT_STREAM
                        Url of output rtsp stream
  --model_name MODEL_NAME
                        Name of the model
  --verbose VERBOSE     Should client dump debug information
  --input_name INPUT_NAME
                        Name of the model's input
```

- Usage example

```bash
docker run -v $(pwd):/workspace rtsp_client --grpc_address localhost:9000 --input_stream 'rtsp://localhost:8080/channel1' --output_stream 'rtsp://localhost:8080/channel2'
```

Then read rtsp stream using ffplay

```bash
ffplay -pix_fmt yuv420p -video_size 704x704 -rtsp_transport tcp rtsp://localhost:8080/channel2
```
