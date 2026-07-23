# ByteTrack Demo Setup

## 1. Download the YOLOX Tiny ONNX Model

Download the YOLOX Tiny ONNX model from the official release:

https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx

---

## 2. Convert the ONNX Model to TensorFlow Lite

Open a Google Colab notebook and:

1. Install `onnx2tf`.
2. Upload `yolox_tiny.onnx` to the notebook.
3. Run:

```bash
!onnx2tf -i yolox_tiny.onnx -o yolox_tiny
```

This generates the TensorFlow Lite model.

---

## 3. Download COCO Class Labels

Download the COCO 80-class label file:

https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_80cl.txt

---

# Running the Demo

## 1. Start the OpenVINO Model Server

```bash
docker run -d \
    -v $PWD:/demo \
    -p 9000:9000 \
    openvino/model_server:latest \
    --config_path /demo/config.json \
    --port 9000
```

---

## 2. Create an RTSP Input Stream

Use FFmpeg to publish your webcam as an RTSP stream.

```bash
ffmpeg -f dshow -video_size 1280x720 \
-i video="HP True Vision FHD Camera" \
-f rtsp -rtsp_transport tcp \
rtsp://localhost:8554/channel1
```

> **Work in Progress:** The following H.264-based streaming command is still being evaluated and may not work correctly in all setups.

```bash
ffmpeg -f dshow \
-video_size 1280x720 \
-framerate 30 \
-i video="HP True Vision FHD Camera" \
-c:v libx264 \
-crf 18 \
-preset veryfast \
-f rtsp \
-rtsp_transport tcp \
rtsp://localhost:8554/channel1
```

---

## 3. Run the Real-Time Stream Analysis Client

```bash
python client.py \
    --grpc_address localhost:9000 \
    --input_stream rtsp://localhost:8554/channel1 \
    --output_stream rtsp://localhost:8554/channel2 \
    --model_name ByteTrack \
    --input_name input_video
```

---

## 4. View the Output Stream

**Option 1 (recommended):**

```bash
ffplay -rtsp_transport tcp \
-vf "scale=704:704,format=yuv420p" \
rtsp://localhost:8554/channel2
```

**Option 2 (verbose logging):**

```bash
ffplay -loglevel verbose \
-rtsp_transport tcp \
rtsp://localhost:8554/channel2
```
