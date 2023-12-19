git clone https://github.com/hilliao/model_server.git

# follow the steps at https://github.com/hilliao/model_server/tree/main/demos/real_time_stream_analysis/python/use_cases/person_vehicle_bike_detection
# to download the prebuilt models
mkdir -p workspace/person-vehicle-bike-detection-2002/1
wget -P workspace/person-vehicle-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.bin
wget -P workspace/person-vehicle-bike-detection-2002/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/person-vehicle-bike-detection-2002/FP32/person-vehicle-bike-detection-2002.xml

# run the model server in daemon mode
docker run --name person-vehicle-bike-detection -d -v $PWD/workspace/person-vehicle-bike-detection-2002:/model -p 9000:9000 openvino/model_server:latest --model_path /model --model_name person-vehicle-bike-detection --layout NHWC:NCHW --shape auto --port 9000

# create the python virtual env
mkdir -p py-venv
python3 -m venv py-venv/
. py-venv/bin/activate
pip3 install -r model_server/demos/real_time_stream_analysis/python/requirements.txt

# execute the person detection use case
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/secrets/person-detection-logger@test-vpc-341000.iam.gserviceaccount.com
export PERSON_DETECTION_MIN_LOG_INTERVAL_SECONDS=5
export PERSON_DETECTION_DEBUG=1
export PERSON_DETECTION_GCS_BUCKET=
export PERSON_DETECTION_GCS_FOLDER=

python3 model_server/demos/real_time_stream_analysis/python/real_time_stream_analysis.py --stream_url rtsp://$USERNAME:$PASSWORD@192.168.1.42:554  --ovms_url localhost:9000 --model_name person-vehicle-bike-detection  --visualizer_port 9001
