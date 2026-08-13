# ByteTrack Demo Setup

End-to-end demo: video source (webcam / file / RTSP) → OpenVINO Model Server (YOLOX Tiny+ ByteTrack) → output (screen / file / RTSP).

---

## 1. Model Preparation

The detector stage of the pipeline runs on [OpenVINO](https://github.com/openvinotoolkit/openvino)-optimized YOLOX models. The following FP16 variants are currently supported:

| Model | HuggingFace Repo |
|---|---|
| YOLOX-Tiny (fp16 precision)| `OpenVINO/yolox_tiny-fp16-ov` |
| YOLOX-Tiny (int8 precision)| `OpenVINO/yolox_tiny-int8-ov` |

`yolox_tiny-fp16-ov` is the default used in this demo.

### Install requirements

```bash
pip install -r requirements.txt
```

### Download a model

```bash
python download_models.py --model OpenVINO/yolox_tiny-fp16-ov
```

> Swap `--model` for any of the four repo IDs listed above to use a different YOLOX size. The script also downloads the COCO class list used for labeling detections.

This populates the local model directory that `config.json` (used by the OpenVINO Model Server in step 3) points to, and that ByteTrack consumes downstream for tracking.

---

## 2. Start an RTSP relay server

If you don't already have one running, start MediaMTX (or an equivalent RTSP server) so the streams below have somewhere to publish to:

```bash
docker run --rm -d -p 8554:8554 -e RTSP_PROTOCOLS=tcp bluenviron/mediamtx:latest
```

> Only needed if you plan to use RTSP input and/or output (see Demo 3 below). It isn't required for the local webcam→screen or video-file→mp4 demos.

---

## 3. Start the OpenVINO Model Server

```bash
docker run -d -v $PWD:/demo -p 9000:9000 openvino/model_server:latest --config_path /demo/config.json --port 9000
```

---

## 4. Run the Demo

The `client.py` script (from `real_time_stream_analysis`) supports several combinations of input and output, so the same server can be exercised in different ways depending on what you have available.

### Demo A — Local webcam → screen

Reads directly from a local camera and renders the tracked output in a window.

```bash
python client.py --grpc_address localhost:9000 --input_stream 0 --output_stream screen
```

- `--input_stream 0` — camera device ID `0` (use `1`, `2`, etc. for additional cameras).
- `--output_stream screen` — opens a live preview window instead of writing to a file or stream.

### Demo B — Video file → video file

Reads from an encoded video file and writes the annotated result to a new video file.

```bash
curl -L "https://raw.githubusercontent.com/FoundationVision/ByteTrack/main/videos/palace.mp4" -o video.mp4
python client.py --grpc_address localhost:9000 --input_stream video.mp4 --output_stream output.mp4
```

- `--input_stream video.mp4` — path to the source video.
- `--output_stream output.mp4` — path where the tracked/annotated video is saved.

### Demo C — RTSP → RTSP

Full end-to-end streaming demo: publish a webcam feed to an RTSP endpoint, run detection + tracking on it, and publish the annotated result to a second RTSP endpoint.

**1. Publish your webcam as an RTSP input stream**

```bash
ffmpeg -f dshow -video_size 1280x720 -i video="HP True Vision FHD Camera" -f rtsp -rtsp_transport tcp rtsp://localhost:8554/channel1
```

**2. Run the client against the RTSP input/output**

```bash
python client.py --grpc_address localhost:9000 --input_stream rtsp://localhost:8554/channel1 --output_stream rtsp://localhost:8554/channel2 --model_name ByteTrack --input_name input_video
```

**3. View the output stream**

Option 1 (recommended):

```bash
ffplay -rtsp_transport tcp -vf "scale=704:704,format=yuv420p" rtsp://localhost:8554/channel2
```

Option 2 (verbose logging):

```bash
ffplay -loglevel verbose -rtsp_transport tcp rtsp://localhost:8554/channel2
```

---

## Summary of I/O Options

| Demo | Input | Output | RTSP server required? |
|---|---|---|---|
| A | Local webcam (`0`) | `screen` | No |
| B | Video file (`video.mp4`) | Video file (`output.mp4`) | No |
| C | RTSP stream | RTSP stream | Yes |