# Exemplary Use Case - Person, Vehicle, Bike Detection

## Download Model

Model used in this example is [person-vehicle-bike-detection-2002](https://docs.openvino.ai/2022.1/omz_models_model_person_vehicle_bike_detection_2002.html).
Create `workspace` and model directory and download model in IR format:
```
mkdir -p workspace/person-vehcile-bike-detection-2002/1
wget -P workspace/person-vehcile-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.bin
wget -P workspace/person-vehcile-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.xml
```

## Run OpenVINO Model Server

Once you have the model, launch OpenVINO Model Sever and mount model catalog to the container:

```
 docker run --rm -v $PWD/workspace/person-vehcile-bike-detection-2002:/model -p 9000:9000 openvino/model_server:latest --model_path /model --model_name person-vehicle-bike-detection --layout NHWC:NCHW --port 9000 
```

## Switch Use Case used for pre and post processing

Modify streaming app main script - [`real_time_stream_analysis.py`](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/real_time_stream_analysis/python/real_time_stream_analysis.py) to contain the following:

```
from use_cases import PersonVehicleBikeDetection

...

io_processor = IOProcessor(PersonVehicleBikeDetection, visualizer_frames_queue)
```

## Run Stream Analysis

As this use case implements only visualization in post processing, run with visualizer:

```
python3 real_time_stream_analysis.py --stream_url <rtsp_stream_url> --ovms_url localhost:9000 --model_name person-vehicle-bike-detection --visualizer_port 5000
```

## Example Browser Preview

<img src="https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/real_time_stream_analysis/python/assets/visualizer_example_browser.gif">