# Real Time Stream Analysis Demo {#ovms_demo_real_time_stream_analysis}
## Overview

For object classification, detection and segmentation we use CV (Computer Vision) models that take visual data on the input and return predictions like classification results, bounding boxes parameters etc. By visual data we often mean video stream generated in real time by different kinds of cameras. 

In this demo you'll see how to analyze RTSP (Real Time Streaming Protocol) stream using OpenVINO Model Server for inference.

![concept](assets/concept.jpg)

The stream analysis app is started with `rtsp_client.py` script. It reads frames from the provided stream URL, runs pre and post processing and requests inference on specified model served by OVMS.

Specific use case actions are defined in use case implementation - some notifications could be sent to another service if object of interest has been detected etc.

As part of postprocessing, inference results can be visualized. The demo can optionally start Flask server that will host inference preview as defined for the use case. 


## Prerequisites

In order to make this demo work you need to:
- use Python 3.7+
- have access to live RTSP stream
- have access to OpenVINO Model Server with your model of choice deployed
- have a use case implementation

The stream analysis app needs to have access to RTSP stream to read from and OVMS to run inference on. Apart from that you need use case implementation that defines pre and postprocessing. Some exemplary use cases are available in [use cases catalog](https://github.com/openvinotoolkit/model_server/blob/main/demos/mediapipe).

## Start the real time stream analysis

Mediapipe graph can be used for remote analysis of individual images but the client can use it for a complete video stream processing.
Below is an example how to run a client reading encoded rtsp video stream.

![rtsp](rtsp.png)

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

(Optionally) Build the docker image with the python client for video stream reading an remote analysis:
```
docker build ../../common/stream_client/ -t rtsp_client
```

Or install python dependencies directly
```bash
pip3 install -r ../../common/stream_client/requrements.txt
```

### Start the client

- Command

```bash
python3 rtsp_client.py --help
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
python3 rtsp_client.py --grpc_address localhost:9000 --input_stream 'rtsp://localhost:8080/channel1' --output_stream 'rtsp://localhost:8080/channel2'
```

Then read rtsp stream using ffplay

```bash
ffplay -pixel_format yuv420p -video_size 704x704 -rtsp_transport tcp rtsp://localhost:8080/channel2
```

### Inference using prerecorded video

One might as well use prerecorded video and schedule it for inference.
Replace video.mp4 with your video file.

```bash
python3 rtsp_client.py --grpc_address localhost:9000 --input_stream 'workspace/video.mp4' --output_stream 'workspace/output.mp4'
```
