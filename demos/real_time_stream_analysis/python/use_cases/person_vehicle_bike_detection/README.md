# Exemplary Use Case - Person, Vehicle, Bike Detection

## Download Model

Model used in this example is [person-vehicle-bike-detection-2002](https://docs.openvino.ai/2023.2/omz_models_model_person_vehicle_bike_detection_2002.html).
Create `workspace` and model directory and download model in IR format:
```bash
mkdir -p workspace/person-vehicle-bike-detection-2002/1
wget -P workspace/person-vehicle-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.bin
wget -P workspace/person-vehicle-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.xml
```

## Run OpenVINO Model Server

Once you have the model, launch OpenVINO Model Sever and mount model catalog to the container:
To run the Docker container in detached mode, replace `--rm` with `-d`
```bash
docker run --rm --name person-vehicle-bike-detection -v $PWD/workspace/person-vehicle-bike-detection-2002:/model -p 9000:9000 openvino/model_server:latest --model_path /model --model_name person-vehicle-bike-detection --layout NHWC:NCHW --shape auto --port 9000 
```

## Switch Use Case used for pre and post processing

Modify streaming app main script - [`real_time_stream_analysis.py`](https://github.com/openvinotoolkit/model_server/blob/main/demos/real_time_stream_analysis/python/real_time_stream_analysis.py) to contain the following:

```
from use_cases import PersonVehicleBikeDetection

...

io_processor = IOProcessor(PersonVehicleBikeDetection, visualizer_frames_queue)
```

## Run Stream Analysis
Refer to `run.sh` for configuring a Python virtual environment to install the required dependencies.

As this use case implements only visualization in post processing, run with visualizer:

```
python3 real_time_stream_analysis.py --stream_url <rtsp_stream_url> --ovms_url localhost:9000 --model_name person-vehicle-bike-detection --visualizer_port 5000
```

## Example Browser Preview

<img src="https://github.com/openvinotoolkit/model_server/blob/main/demos/real_time_stream_analysis/python/assets/visualizer_example_browser.gif">

## Person detection
A use case is writing detected person to local files at $HOME/Pictures ; if the path does not exist, files are not written.
Examine the `def postprocess(inference_result: np.ndarray):` method in `person_vehicle_bike_detection.py`

## Debug options
To enable printing debugging info to the terminal, set
- export PERSON_DETECTION_DEBUG=1

To set the log sample rate by interval of 5 seconds, set
- export PERSON_DETECTION_MIN_LOG_INTERVAL_SECONDS=5

Default is 3 seconds. 

## Optional Google Cloud logging and storage integration

Refer to run.sh for the lines that set the following environment variables:
- export GOOGLE_APPLICATION_CREDENTIALS=/file/path/to/service-account.json
- export PERSON_DETECTION_GCS_BUCKET=create-a-bucket-and-grant-SA-storage-object-creator-role
- export PERSON_DETECTION_GCS_FOLDER=a-string-of-the-folder-name
- export PERSON_DETECTION_LOCAL_FOLDER=$HOME/Pictures

When the variables are set, log messages are written to Google Cloud logging. Pictures of Person detected are uploaded
to the cloud storage bucket. Make sure the service account has `storage object creator` IAM role for write to succeed.
The local file is
**gs://create-a-bucket-and-grant-SA-storage-object-creator-role/a-string-of-the-folder-name/detected-person_{formatted_datetime}.png**
where formatted_datetime is `"%Y-%m-%d_%H-%M-%S.%f"` ;