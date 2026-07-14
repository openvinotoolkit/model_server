YOLOX Tiny ONNX download link

https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx

After downloading the onnx model, In google colab

download onnx2tf in colab, move the onnx weights to the notebook and use the following command in colab notebook to convert onnx weights to tflite.

```bash
!onnx2tf -i yolox_tiny.onnx -o yolox_tiny
```

COCO class labels download

https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_80cl.txt

------
Use this command to build container for bytetrack demo
```bash
docker run -d -v $PWD:/demo -p 9000:9000 openvino/model_server:latest --config_path /demo/config.json --port 9000
```

------


------
To create RTSP input channel
```bash
ffmpeg -f dshow -i video="HP True Vision FHD Camera" -f rtsp -rtsp_transport tcp rtsp://localhost:8554/channel1
```


To run real_time_stram_analysis for demo use this command
```bash
python client.py --grpc_address localhost:9000 --input_stream 'rtsp://localhost:8554/channel1' --output_stream 'rtsp://localhost:8554/channel2' --model_name ByteTrack --input_name input_video --ffmpeg_output_width 416 --ffmpeg_output_height 416
```

To run RTSP channel2 output

```bash
ffplay -loglevel verbose -rtsp_transport tcp rtsp://localhost:8554/channel2
```