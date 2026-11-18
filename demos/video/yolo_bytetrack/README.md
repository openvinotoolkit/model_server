# YOLOX + ByteTrack WebRTC POC

This POC uses the served MediaPipe YOLOX-Tiny + ByteTrack graph for live VP8
frames. Each processed frame produces one `oai-events` message with normalized
boxes, tracker IDs, labels, and scores. The existing speech-to-text path remains
independent and can run alongside video.

## Run from the build container

With the checkout mounted at `/ovms`, run this in the OVMS build container:

```bash
cd /ovms
./run_yolo_bytetrack.sh
```

The root-level shortcut uses this machine's host IP (`10.123.247.3`) for WebRTC
media, matching `run_voxtral.sh`. To use another address, set
`OVMS_WEBRTC_ANNOUNCED_IP` before running it. To pass an explicit IP without
using the shortcut, call `demos/video/yolo_bytetrack/run_server.sh SERVER_IP`.

The script builds `//src:ovms` with WebRTC enabled, downloads and prepares the
YOLOX-Tiny model and COCO labels on the first run, then starts OVMS. Model
assets are stored under `/ovms/models/yolo_bytetrack`; set `OVMS_MODEL_DIR` to
change that location. It generates a runtime config there so the model and
label-map paths work inside the build container without changing the checked-in
graph.

When model preparation is needed, the script creates a `.venv` under the model
directory, reuses available system packages (including the builder's OpenVINO),
and installs missing `openvino` / `huggingface_hub` dependencies there. It does
not modify the container's system Python packages.

The first argument is the IP address clients can reach for WebRTC media. Use
`127.0.0.1` when testing from the same host, or the build container host's IP
for remote clients. The optional second argument is the browser page's origin;
it defaults to `http://localhost:8000`. `REST_PORT` defaults to `6666`,
`OVMS_WEBRTC_PORT_RANGE` defaults to `52000-52100`, and `BAZEL_JOBS` defaults to
`8`. The build container must publish the REST port and UDP media range.

In another terminal on the host, serve the demo page from the checkout:

```bash
cd /path/to/model_server
python3 -m http.server 8000 --bind 127.0.0.1 \
    --directory demos/video/yolo_bytetrack
```

Open `http://localhost:8000`, set the OVMS REST URL to
`http://localhost:6666`, and click **Start**. Camera access requires localhost
or HTTPS.

## Browser client

For connection steps, communication channels, and the tracking-event schema,
see the [web client integration guide](CLIENT.md).

Select the audio checkbox and set the model to a served speech-to-text graph to
run ASR simultaneously; the video graph continues to come from
`OVMS_WEBRTC_VIDEO_GRAPH`.

The video event is a POC extension, not an OpenAI-standard video event:

```json
{
  "type": "conversation.item.input_video_tracking.completed",
  "event_id": "event_20",
  "item_id": "video_3_13",
  "frame": 13,
  "rtp_timestamp": 3715954780,
  "width": 640,
  "height": 480,
  "objects": [
    {
      "id": 4,
      "label": "person",
      "score": 0.91,
      "box": {"x": 0.12, "y": 0.20, "w": 0.30, "h": 0.60}
    }
  ],
  "transcript": "person#4 [0.12,0.20,0.30,0.60]"
}
```

Boxes are top-left coordinates relative to the original decoded frame. An event
with an empty `objects` array is emitted for frames where no tracks are active.
The WebRTC decode queue and live graph queue are bounded; frames are dropped
under backpressure rather than blocking the media transport thread.

## aiortc smoke test

With OVMS running with this graph and a short video containing visible people:

```bash
OVMS_YOLO_URL=http://127.0.0.1:6666 \
WEBRTC_YOLO_VIDEO_FILE=/path/to/people.mp4 \
    tests/webrtc_e2e/.venv/bin/python -m pytest -v tests/webrtc_e2e/test_webrtc_yolo_bytetrack.py
```

The test verifies normalized boxes, valid scores, consecutive tracker-ID
persistence, and the absence of error events. It uses
`OVMS_WEBRTC_VIDEO_GRAPH_NAME` if the server graph name differs from the
default `YoloByteTrackWebRTC`.
