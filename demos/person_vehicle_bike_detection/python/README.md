# Person, vehicle, bike detection with multiple data sources {#ovms_demo_person_vehicle_bike_detection}

The purpose of this demo is to show how to send data from multiple sources (cameras, video files) to a model served in OpenVINO Model Server.

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip 

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

## Prepare model files
```console
curl --create-dir https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.bin -o model/1/person-vehicle-bike-detection-crossroad-0078.bin

curl --create-dir https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-crossroad-0078/FP32/person-vehicle-bike-detection-crossroad-0078.xml -o model/1/person-vehicle-bike-detection-crossroad-0078.xml
```

## Server Deployment
:::{dropdown} **Deploying with Docker**
```bash
docker run -d -v `pwd`/model:/models -p 9000:9000 openvino/model_server:latest --model_path /models --model_name person-vehicle-detection --port 9000 --shape auto
```
:::
:::{dropdown} **Deploying on Bare Metal**
Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.
```bat
cd demos\person_vehicle_bike_detection\python
ovms --model_path model --model_name person-vehicle-detection --port 9000 --shape auto
```
:::
## Running the client application

```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/person_vehicle_bike_detection/python
pip3 install -r requirements.txt
python person_vehicle_bike_detection.py --help
```

### Arguments

| Argument      | Description |
| :---        |    :----   |
| -h,--help       | Show help message and exit       |
| -n NETWORK_NAME, --network_name NETWORK_NAME   |   Network name      |
| -l INPUT_LAYER, --input_layer INPUT_LAYER | Input layer name |
| -o OUTPUT_LAYER, --output_layer OUTPUT_LAYER | Output layer name |
| -d FRAME_SIZE, --frame_size FRAME_SIZE | Input frame width and height that matches used model |
| -c NUM_CAMERAS, --num_cameras NUM_CAMERAS | Number of cameras to be used |
| -f FILE, --file FILE | Path to the video file |
| -i IP, --ip IP| IP address of the ovms|
| -p PORT, --port PORT | Port of the ovms |

### Using with video file

Copy example video file:
```console
git clone "https://github.com/intel-iot-devkit/sample-videos.git"
```

Set `camera` count to `0` with `-c 0` and provide path to the video file with `-f` parameter.
```console
python person_vehicle_bike_detection.py -n person-vehicle-detection -l data -o detection_out -d 1024 -c 0 -f sample-videos/person-bicycle-car-detection.mp4 -i localhost -p 9000
```
Output:
```bash
[$(levelname)s ] Video0 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Video0 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Video0 fps: 7, Inf fps: 7, dropped fps: 0
```

### Using with video file and camera

Set `camera` count to `1` with `-c 1` and provide path to the video file with `-f` parameter.
```console
python person_vehicle_bike_detection.py -n person-vehicle-detection -l data -o detection_out -d 1024 -c 1 -f sample-videos/person-bicycle-car-detection.mp4 -i localhost -p 9000
```

Console logs:
```bash
[$(levelname)s ] Video1 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Camera0 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Video1 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Camera0 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Video1 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Camera0 fps: 8, Inf fps: 8, dropped fps: 0
```

> **NOTE:** You should also be seeing the GUI showing the video frame and bounding boxes drawn with the detected class name