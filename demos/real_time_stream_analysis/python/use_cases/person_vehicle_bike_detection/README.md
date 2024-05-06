# Exemplary Use Case - Person, Vehicle, Bike Detection

## Download Model

Model used in this example is [person-vehicle-bike-detection-2002](https://docs.openvino.ai/2023.2/omz_models_model_person_vehicle_bike_detection_2002.html).
Create `workspace` and model directory and download model in IR format:
```bash
mkdir -p workspace/person-vehicle-bike-detection-2002/1
wget -P workspace/person-vehicle-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.bin
wget -P workspace/person-vehicle-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.xml
```

## Install the Python packages
Create and activate a Python virtual environment 
```commandline
sudo apt install python3-pip
# for Python 3.12 only
sudo apt install python3.12-venv && \
mkdir py-venv && \
python3 -m venv py-venv && \
. py-venv/bin/activate
```
In the Python virtual environment, Install [requirements.txt](..%2F..%2Frequirements.txt)
```commandline
(py-venv)$ pip3 install -r model_server/demos/real_time_stream_analysis/python/requirements.txt
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
python3 real_time_stream_analysis.py --stream_url <rtsp_stream_url> --ovms_url localhost:9000 --model_name person-vehicle-bike-detection --visualizer_port 9001
```
View the predicted video stream at http://localhost:9001

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

Refer to `run.sh` for the lines that set the following environment variables:
- export GOOGLE_APPLICATION_CREDENTIALS=/file/path/to/service-account.json
- export PERSON_DETECTION_GCS_BUCKET=create-a-bucket-and-grant-SA-storage-object-creator-role
- export PERSON_DETECTION_GCS_FOLDER=a-string-of-the-folder-name
- export PERSON_DETECTION_LOCAL_FOLDER=$HOME/Pictures

When the variables are set, log messages are written to Google Cloud logging. Pictures of Person detected are uploaded
to the cloud storage bucket. Make sure the service account has `storage object creator` IAM role for write to succeed.
The file format is
**gs://$GCS_BUCKET/us-amcrest-cam-0/detected-person_{formatted_datetime}.png**
where formatted_datetime is `"%Y-%m-%d_%H-%M-%S.%f"` ;

## Deploy to production
Deploy the following BASH script partially from `run.sh` to /usr/local/bin/person-detection.sh for execution.
You can also configure the person-vehicle-bike-detection docker container to
[auto restart](https://docs.docker.com/config/containers/start-containers-automatically/).

0. If GOOGLE_APPLICATION_CREDENTIALS exists, logging to Google Cloud is enabled with
`PERSON_DETECTION_GCP_LOG_NAME` environment variable. Ensure the service account has logs writer IAM role.
1. If PERSON_DETECTION_GCS_BUCKET, PERSON_DETECTION_GCS_FOLDER environment variables exist, files saved to `PERSON_DETECTION_LOCAL_FOLDER` is uploaded to Google Cloud storage bucket. Ensure the service account has `storage object creator` IAM role in the bucket.
2. Executing the script and leave it to run. To top it, Ctrl+C may not terminate properly. Try Ctrl+Z, `ps` to see which process is python3. Kill it with `kill -9 [PID_of_Python3]`.

Replace the environment variables such as PROJECT_ID in the following script.
```commandline
export GOOGLE_APPLICATION_CREDENTIALS=/home/hil/Encfs/confidential/secrets/person-detection-logger@$PROJECT_ID.iam.gserviceaccount.com
export PERSON_DETECTION_CAMERA_ID=us-amcrest-cam-0
export PERSON_DETECTION_CONFIDENCE_THRESHOLD=0.9
export PERSON_DETECTION_DEBUG=1
export PERSON_DETECTION_GCP_LOG_NAME=person-detection
export PERSON_DETECTION_GCS_BUCKET=$GCS_BUCKET
export PERSON_DETECTION_GCS_FOLDER=us-amcrest-cam-0
export PERSON_DETECTION_LOCAL_FOLDER=/mnt/1tb/ftp/ipcam/autodelete/person-detection
export PERSON_DETECTION_MIN_LOG_INTERVAL_SECONDS=10

mkdir -p -v $PERSON_DETECTION_LOCAL_FOLDER

/home/hil/git/model_server/py-venv/bin/python3 \
/home/hil/git/model_server/demos/real_time_stream_analysis/python/real_time_stream_analysis.py \
--stream_url rtsp://$IPCAM_USERNAME:$IPCAM_PASSWORD@192.168.1.42:554  --ovms_url localhost:9000 --model_name person-vehicle-bike-detection  --visualizer_port 9001
```
Be careful that analyzed video stream at http://localhost:9001 is not password protected.
