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

from jinja2 import Template
from data.performance_constants import DATASET, OVMS_DATASET, AMS_PORT, OVMS_PORT_FOR_AMS, \
    MODEL_PATH_FOR_OVMS, OVMS_PORT, CONFIG_PATH, PARAMS, DLDT_PACKAGE, CONF_PATH_OFFCIAL, \
    AMS_START_SCRIPT_PATH, AMS_START_SCRIPT_PATH_OFFCIAL


def read_and_rewrite_file(filename: str, **kwargs):
    with open(filename) as file:
        template = Template(file.read())

    with open(filename, 'w+') as file:
        file.write(template.render(**kwargs))


def prepare_image_ams(nireq, plugin_config, grpc_workers):
    copy_config_file(nireq=nireq, plugin_config=plugin_config)
    edit_ams_start_script(grpc_workers=grpc_workers)
    cmd = ["make", "docker_build_ams", "DLDT_PACKAGE_URL={}".format(DLDT_PACKAGE)]
    subprocess.run(cmd)


def copy_config_file(nireq, plugin_config):
    # adjust new config file with proper params
    kwargs = {"nireq": nireq, "plugin_config": plugin_config}
    read_and_rewrite_file(CONFIG_PATH, **kwargs)

    # backup previous config file
    cmd = ["mv", CONF_PATH_OFFCIAL, "{}.backup".format(CONF_PATH_OFFCIAL)]
    subprocess.run(cmd)

    # move new config file
    cmd = ["mv", CONFIG_PATH, CONF_PATH_OFFCIAL]
    subprocess.run(cmd)


def edit_ams_start_script(grpc_workers):
    # adjust new script file with proper params
    kwargs = {"grpc_workers": grpc_workers}
    read_and_rewrite_file(AMS_START_SCRIPT_PATH, **kwargs)

    # backup previous script file
    cmd = ["mv", AMS_START_SCRIPT_PATH_OFFCIAL, "{}.backup".format(AMS_START_SCRIPT_PATH_OFFCIAL)]
    subprocess.run(cmd)

    # move new script file
    cmd = ["mv", AMS_START_SCRIPT_PATH, AMS_START_SCRIPT_PATH_OFFCIAL]
    subprocess.run(cmd)


def clean_up():
    # remove adjusted config file and script file
    cmd = ["rm", "-rf", CONF_PATH_OFFCIAL, AMS_START_SCRIPT_PATH_OFFCIAL]
    subprocess.run(cmd)

    # back to previous config file
    cmd = ["mv", "{}.backup".format(CONF_PATH_OFFCIAL), CONF_PATH_OFFCIAL]
    subprocess.run(cmd)

    # back to previous script file
    cmd = ["mv", "{}.backup".format(AMS_START_SCRIPT_PATH_OFFCIAL), AMS_START_SCRIPT_PATH_OFFCIAL]
    subprocess.run(cmd)


@pytest.fixture(scope="function", params=PARAMS)
def ams_and_ovms(request):
    # adjust file with params for ams image build
    prepare_image_ams(request.param["nireq"], request.param["plugin_config"], request.param["grpc_workers"])

    # run ams container
    container_name_ams = "ams_{}_cores".format(request.param["cores"])
    cmd = ["docker", "run", "--cpus={}".format(request.param["cores"]),
           "--name", container_name_ams,
           "-d", "-p", "{}:{}".format(AMS_PORT, AMS_PORT),
           "-p", "{}:{}".format(OVMS_PORT_FOR_AMS, OVMS_PORT_FOR_AMS), "ams", "/ams_wrapper/start_ams.sh",
           "--ams_port={}".format(AMS_PORT), "--ovms_port={}".format(OVMS_PORT_FOR_AMS)]
    subprocess.run(cmd)
    time.sleep(10)

    # run ovms container
    container_name_ovms = "ovms_{}_cores".format(request.param["cores"])
    cmd = ["docker", "run", "--cpus={}".format(request.param["cores"]),
           "-v", "{}:/models".format(MODEL_PATH_FOR_OVMS), "--name", container_name_ovms,
           "-d", "-p", "{}:{}".format(OVMS_PORT, OVMS_PORT), "ie-serving-py:latest", "/ie-serving-py/start_server.sh",
           "ie_serving", "config", "--config_path", CONFIG_PATH,
           "--port", OVMS_PORT, "--grpc_workers", request.param["grpc_workers"]]
    subprocess.run(cmd)
    time.sleep(10)

    def finalizer():
        cmd = ["docker", "rm", "-f", container_name_ams, container_name_ovms]
        subprocess.run(cmd)
        clean_up()

    request.addfinalizer(finalizer)
    return request.param


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
