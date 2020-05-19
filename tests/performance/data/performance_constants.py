#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

# PATHS
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATASET = os.path.join(ROOT_PATH, "tests", "functional", "fixtures", "test_images", "performance")
OVMS_CLIENT = os.path.join(ROOT_PATH, "example_client", "face_detection_performance.py")
OVMS_DATASET = os.path.join(DATASET, "test_dir")
MODEL_PATH_FOR_OVMS = "/opt/models/models/"
CONFIG_PATH = os.path.join(ROOT_PATH, "tests", "performance", "performance_config_ovms.json")
CONFIG_PATH_TEMP = os.path.join(ROOT_PATH, "tests", "performance", "performance_config_ovms_tmp.json")
CONF_PATH_OFFCIAL = os.path.join(ROOT_PATH, "extras", "ams_models", "ovms_config.json")
CONFIG_PATH_INTERNAL = os.path.join("/", "opt", "models", "ovms_config.json")
AMS_START_SCRIPT_PATH_OFFCIAL = os.path.join(ROOT_PATH, "extras", "ams_wrapper", "start_ams.sh")
AMS_START_SCRIPT_PATH = os.path.join(ROOT_PATH, "tests", "performance", "performance_start_ams.sh")
AMS_CLIENT_PATH = os.path.join(ROOT_PATH, "tests", "performance", "ams_client.py")
OVMS_CLIENT_PATH = os.path.join(ROOT_PATH, "tests", "performance", "ovms_client.sh")

# OTHERS PARAMS
ITERATIONS = 1000
AMS_ADDRESS = "localhost"
OVMS_PORT_FOR_AMS = "9000"
AMS_PORT = "5000"
OVMS_PORT = "9007"

# ov package
DLDT_PACKAGE = os.environ["DLDT_PACKAGE"]


# PERFORMANCE PARAMS
NIREQ10_GRPCWORKERS_10_MULTISTREAM_4CORES = {"nireq": 10, "grpc_workers": "10",
                                              "plugin_config": "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}",
                                             "cores": 4}
NIREQ10_GRPCWORKERS_10_MULTISTREAM_32CORES = {"nireq": 10, "grpc_workers": "10",
                                              "plugin_config": "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}",
                                              "cores": 32}
NIREQ10_GRPCWORKERS_10_SINGLESTREAM_4CORES = {"nireq": 10, "grpc_workers": "10",
                                              "plugin_config": "{\"CPU_THROUGHPUT_STREAMS\": \"1\"}", "cores": 4}
NIREQ10_GRPCWORKERS_10_SINGLESTREAM_32CORES = {"nireq": 10, "grpc_workers": "10",
                                              "plugin_config": "{\"CPU_THROUGHPUT_STREAMS\": \"1\"}", "cores": 32}

PARAMS = [NIREQ10_GRPCWORKERS_10_SINGLESTREAM_32CORES, NIREQ10_GRPCWORKERS_10_SINGLESTREAM_4CORES,
          NIREQ10_GRPCWORKERS_10_MULTISTREAM_4CORES, NIREQ10_GRPCWORKERS_10_MULTISTREAM_32CORES]

# PERFORMANCE MODELS
VEHICLE_DETECTION = {"model_name": "vehicleDetection", "width": "672", "height": "384",
                     "model_name_ovms": "vehicle_detection_adas", "dataset": "single_car_small_reshaped.png",
                     "input_name": "data"}
VEHICLE_ATTRIBUTES = {"model_name": "vehicleClassification", "width": "72", "height": "72",
                     "model_name_ovms": "vehicle_attributes", "dataset": "single_car_small_reshaped_72_x_72.png",
                      "input_name": "input"}
FACE_DETECTION = {"model_name": "faceDetection", "width": "672", "height": "384",
                     "model_name_ovms": "face_detection_adas", "dataset": "single_car_small_reshaped.png",
                  "input_name": "data"}
PERSON_DETECTION = {"model_name": "personVehicleBikeDetection", "width": "1024", "height": "1024",
                     "model_name_ovms": "person_vehicle_bike_detection", "dataset": "single_car_large_reshaped.png",
                    "input_name": "data"}
AGE_RECOGNITION = {"model_name": "ageGenderRecognition", "width": "62", "height": "62",
                     "model_name_ovms": "age_gender_recognition", "dataset": "emotions_smile_reshaped.jpg",
                   "input_name": "data"}
EMOTION_RECOGNITION = {"model_name": "emotionsRecognition", "width": "64", "height": "64",
                     "model_name_ovms": "emotions_recognition", "dataset": "emotions_smile.jpg",
                       "input_name": "data"}

MODELS = [VEHICLE_DETECTION, VEHICLE_ATTRIBUTES, FACE_DETECTION, PERSON_DETECTION, AGE_RECOGNITION, EMOTION_RECOGNITION]
