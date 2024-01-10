# MediaPipe Holistic Demo {#ovms_docs_demo_mediapipe_holistic}

This guide shows how to implement [MediaPipe](../../../docs/mediapipe.md) graph using OVMS.

Example usage of graph that accepts Mediapipe::ImageFrame as a input:

The demo is based on the [upstream Mediapipe holistic demo](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md).

## Prepare the server deployment

Clone the repository and enter mediapipe object_detection directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/mediapipe/holistic_tracking

./prepare_server.sh

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
docker run -d -v $PWD/mediapipe:/mediapipe -v $PWD/ovms:/models -p 9000:9000 openvino/model_server:latest --config_path /models/config_holistic.json --port 9000
```

## Run client application for holistic tracking - default demo
```bash
pip install -r requirements.txt
# download a sample image for analysis
curl -kL -o girl.jpeg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg
echo "girl.jpeg" > input_images.txt
# launch the client
python mediapipe_holistic_tracking.py --grpc_port 9000 --images_list input_images.txt
Running demo application.
Start processing:
        Graph name: holisticTracking
(640, 960, 3)
Iteration 0; Processing time: 131.45 ms; speed 7.61 fps
Results saved to :image_0.jpg
```
## Output image
![output](output_image.jpg)


## RTSP Client
Mediapipe graph can be used for remote analysis of individual images but the client can use it for a complete video stream processing.
Below is an example how to run a client reading encoded rtsp video stream.

![rtsp](rtsp.png)

Build docker image containing rtsp client along with its dependencies
The rtsp client app needs to have access to RTSP stream to read from and write to.

Example rtsp server [mediamtx](https://github.com/bluenviron/mediamtx)

```bash
docker run --rm -d -p 8080:8554 -e RTSP_PROTOCOLS=tcp bluenviron/mediamtx:latest
```

Then write to the server using ffmpeg, example using video or camera

```bash
ffmpeg -stream_loop -1 -i ./video.mp4 -f rtsp -rtsp_transport tcp rtsp://localhost:8080/channel1
```

```
ffmpeg -f dshow -i video="HP HD Camera" -f rtsp -rtsp_transport tcp rtsp://localhost:8080/channel1
```

Build the docker image with the python client for video stream reading an remote analysis:
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
                      [--model_name MODEL_NAME] [--input_name INPUT_NAME]
                      [--verbose] [--benchmark]
                      [--limit_stream_duration LIMIT_STREAM_DURATION]
                      [--limit_frames LIMIT_FRAMES]

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
  --input_name INPUT_NAME
                        Name of the model's input
  --verbose             Should client dump debug information
  --benchmark           Should client collect processing times
  --limit_stream_duration LIMIT_STREAM_DURATION
                        Limit how long client should run
  --limit_frames LIMIT_FRAMES
                        Limit how many frames should be processed
```

- Usage example

### Inference using RTSP stream

```bash
docker run --network="host" -v $(pwd):/workspace rtsp_client --grpc_address localhost:9000 --input_stream 'rtsp://localhost:8080/channel1' --output_stream 'rtsp://localhost:8080/channel2'
```

Then read rtsp stream using ffplay

```bash
ffplay -pixel_format yuv420p -video_size 704x704 -rtsp_transport tcp rtsp://localhost:8080/channel2
```

### Inference using prerecorded video

One might as well use prerecorded video and schedule it for inference.
Replace horizontal_text.mp4 with your video file.

```bash
docker run --network="host" -v $(pwd):/workspace rtsp_client --grpc_address localhost:9000 --input_stream 'workspace/video.mp4' --output_stream 'workspace/output.mp4'
```
