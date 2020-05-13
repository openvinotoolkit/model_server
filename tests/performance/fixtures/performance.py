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
import subprocess
import pytest
import time


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATASET = os.path.join(ROOT_PATH, "tests", "functional", "fixtures", "test_images", "performance")
OVMS_DATASET = os.path.join(DATASET, "test_dir")

CORES_NUMBERS = [4, 32]

ITERATIONS = 1000
SERVICES = []


@pytest.fixture(scope="function", params=CORES_NUMBERS)
def ams(request):
    container_name = "ams_{}_cores".format(request.param)
    cmd = ["docker", "run", "--cpus={}".format(request.param),
           "--name", container_name,
           "-d", "-p", "5000:5000", "-p", "9000:9000", "ams", "/ams_wrapper/start_ams.sh",
           "--ams_port=5000", "--ovms_port=9000"]
    subprocess.run(cmd)
    time.sleep(10)

    def finalizer():
        cmd = ["docker", "rm", "-f", container_name]
        subprocess.run(cmd)

    request.addfinalizer(finalizer)


@pytest.fixture(scope="function", params=CORES_NUMBERS)
def ovms(request):
    container_name = "ovms_{}_cores".format(request.param)
    cmd = ["docker", "run", "--cpus={}".format(request.param),
           "-v", "/opt/models/ovms/vehicle_detection:/vehicle_detection", "--name", container_name,
           "-d", "-p", "9002:9002", "ie-serving-py:latest", "/ie-serving-py/start_server.sh", "ie_serving",
           "model", "--model_path", "/vehicle_detection", "--model_name", "vehicle_detection", "--port", "9002",
           "--grpc_workers", "10", "--nireq", "10"]
    subprocess.run(cmd)
    time.sleep(10)

    def finalizer():
        cmd = ["docker", "rm", "-f", container_name]
        subprocess.run(cmd)

    request.addfinalizer(finalizer)


@pytest.fixture(scope="function")
def prepare_dataset_for_ovms(request):
    cmd = ["mkdir", OVMS_DATASET]
    subprocess.run(cmd)
    path = os.path.dirname(DATASET)

    cmd = ["cp", os.path.join(path, "single_car_small.png"),
           os.path.join(OVMS_DATASET, "single_car_small.png")]
    subprocess.run(cmd)

    def finalizer():
        cmd = ["rm", "-rf", OVMS_DATASET]
        subprocess.run(cmd)

    request.addfinalizer(finalizer)
