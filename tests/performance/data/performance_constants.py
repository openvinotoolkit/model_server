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

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATASET = os.path.join(ROOT_PATH, "tests", "functional", "fixtures", "test_images", "performance")
OVMS_CLIENT = os.path.join(ROOT_PATH, "example_client", "face_detection_performance.py")
OVMS_DATASET = os.path.join(DATASET, "test_dir")
IMAGE = "single_car_small_reshaped.png"
MODEL_PATH_FOR_OVMS = "/opt/models/ovms"
CONFIG_PATH = os.path.join(ROOT_PATH, "tests", "performance", "performance_config_ovms.json")
CONF_PATH_OFFCIAL = os.path.join(ROOT_PATH, "extras", "ams_models", "ovms_config.json")
AMS_START_SCRIPT_PATH_OFFCIAL = os.path.join(ROOT_PATH, "extras", "ams_wrapper", "start_ams.sh")
AMS_START_SCRIPT_PATH = os.path.join(ROOT_PATH, "tests", "performance", "performance_start_ams.sh")

AMS_CLIENT_PATH = os.path.join(ROOT_PATH, "tests", "performance", "ams_client.py")
OVMS_CLIENT_PATH = os.path.join(ROOT_PATH, "tests", "performance", "ovms_client.sh")

ITERATIONS = 1000
AMS_ADDRESS = "localhost"
OVMS_PORT_FOR_AMS = 9000
AMS_PORT = 5000
OVMS_PORT = 9007
DLDT_PACKAGE = "http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120_online.tgz"

NIREQ10_GRPCWORKERS_10_SINGLESTREAM_4CORES = {"nireq": 10, "grpc_workers": 10, "plugin_config": None, "cores": 4}
PARAMS = [NIREQ10_GRPCWORKERS_10_SINGLESTREAM_4CORES]

VEHICLE_DETECTION = {"model_name": "vehicleDetection", "width": 672, "height": 383}

MODELS = [VEHICLE_DETECTION]
