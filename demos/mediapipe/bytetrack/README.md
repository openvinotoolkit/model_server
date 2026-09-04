# ByteTrack Demo Setup

End-to-end demo: video source (webcam / file) → OpenVINO Model Server (YOLOX Tiny + ByteTrack) → output (screen / file).

---

## Steps

### 1. Clone the repository

Clone the repository, switch to the `gsoc_bytetrack` branch, and move into the demo directory:

```bash
git clone https://github.com/Vishwa2684/model_server
cd model_server
git checkout gsoc_bytetrack
cd demos/mediapipe/bytetrack
```

### 2. Install requirements

Install all the Python dependencies needed by the client and the model download script. Run this from inside the `demos/mediapipe/bytetrack` directory:

```bash
pip install -r requirements.txt
```

### 3. Download a model

Download the detector model that the OpenVINO Model Server will use. This same command also fetches the COCO class list used for labeling detections:

```bash
python download_models.py --model OpenVINO/yolox_tiny-fp16-ov
```

> Swap `--model-repo` for any of the repo IDs listed below to use a different YOLOX size.

| Model | HuggingFace Repo |
|---|---|
| YOLOX-Tiny (fp16 precision)| `OpenVINO/yolox_tiny-fp16-ov` |
| YOLOX-Tiny (int8 precision)| `OpenVINO/yolox_tiny-int8-ov` |

`yolox_tiny-fp16-ov` is the default used in this demo.

This step populates the local model directory that `config.json` (used by the OpenVINO Model Server in step 4) points to, and that ByteTrack consumes downstream for tracking.

### 4. Start the OpenVINO Model Server

Bring up the OpenVINO Model Server as a Docker container. This mounts your current directory into the container so it can read `config.json`, and exposes port 9000 for the client to connect to:

```bash
docker run -d -v $PWD:/demo -p 9000:9000 openvino/model_server:latest --config_path /demo/config.json --port 9000
```

Leave this container running in the background — the client in the next step connects to it over gRPC.

### 5. Run the demo — local webcam → screen

With the model server running, run the client script. This reads directly from your local webcam, runs it through detection + ByteTrack tracking, and renders the annotated output live in a window on your screen:

```bash
cd ../../real_time_stream_analysis/python
python client.py --grpc_address localhost:9000 --input_stream 0 --output_stream screen --model_name ByteTrack --input_name input_video
```

- `--grpc_address localhost:9000` — address of the OpenVINO Model Server started in step 4.
- `--input_stream 0` — camera device ID `0` (use `1`, `2`, etc. if you have multiple cameras and want a different one).
- `--output_stream screen` — opens a live preview window instead of writing to a file or stream.

A window should open showing your webcam feed with tracked bounding boxes drawn on it in real time. To use different input and output streams for real time. Read the documentation on [real time stream analysis](../../real_time_stream_analysis/python/README.md)