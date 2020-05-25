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
import time

from jinja2 import Template
from data.performance_constants import DATASET, OVMS_DATASET, AMS_PORT, OVMS_PORT_FOR_AMS, \
    MODEL_PATH_FOR_OVMS, OVMS_PORT, CONFIG_PATH, DLDT_PACKAGE, CONF_PATH_OFFCIAL, \
    AMS_START_SCRIPT_PATH, AMS_START_SCRIPT_PATH_OFFCIAL, CONFIG_PATH_INTERNAL, CONFIG_PATH_TEMP


def read_and_rewrite_file(filename: str, filename_to_save: str, **kwargs):
    with open(filename) as file:
        template = Template(file.read())

    with open(filename_to_save, 'w+') as file:
        file.write(template.render(**kwargs))


def prepare_image_ams(nireq, plugin_config, grpc_workers):
    copy_config_file_ams(nireq=nireq, plugin_config=plugin_config)
    edit_ams_start_script(grpc_workers=grpc_workers)
    cmd = ["make", "docker_build_ams", "DLDT_PACKAGE_URL={}".format(DLDT_PACKAGE)]
    subprocess.run(cmd)


def copy_config_file_ovms(nireq, plugin_config):

    # adjust new config file with proper params
    kwargs = {"nireq": nireq, "plugin_config": plugin_config}
    read_and_rewrite_file(CONFIG_PATH, CONFIG_PATH_TEMP, **kwargs)

    # copy config for ovms
    new_path = os.path.join(MODEL_PATH_FOR_OVMS, "ovms_config.json")
    cmd = ["cp", CONFIG_PATH_TEMP, new_path]
    subprocess.run(cmd)


def copy_config_file_ams(nireq, plugin_config):
    # adjust new config file with proper params
    kwargs = {"nireq": nireq, "plugin_config": plugin_config}
    read_and_rewrite_file(CONFIG_PATH, CONFIG_PATH_TEMP, **kwargs)

    # copy new config file
    cmd = ["cp", CONFIG_PATH_TEMP, CONF_PATH_OFFCIAL]
    subprocess.run(cmd)


def edit_ams_start_script(grpc_workers):
    # adjust new script file with proper params
    kwargs = {"grpc_workers": int(grpc_workers)}
    read_and_rewrite_file(AMS_START_SCRIPT_PATH, AMS_START_SCRIPT_PATH, **kwargs)

    # copy new script file
    cmd = ["cp", AMS_START_SCRIPT_PATH, AMS_START_SCRIPT_PATH_OFFCIAL]
    subprocess.run(cmd)


def run_ams(param):
    # adjust file with params for ams image build
    prepare_image_ams(param["nireq"], param["plugin_config"], param["grpc_workers"])

    # run ams container
    container_name_ams = "ams_{}_cores".format(param["cores"])
    cmd = ["docker", "run", "--cpus={}".format(param["cores"]),
           "--name", container_name_ams,
           "-d", "-p", "{}:{}".format(AMS_PORT, AMS_PORT),
           "-p", "{}:{}".format(OVMS_PORT_FOR_AMS, OVMS_PORT_FOR_AMS), "ams", "/ams_wrapper/start_ams.py",
           "--ams_port={}".format(AMS_PORT), "--ovms_port={}".format(OVMS_PORT_FOR_AMS)]
    subprocess.run(cmd)
    time.sleep(10)

    return container_name_ams


def cleanup_ams(container_name_ams):
    cmd = ["docker", "rm", "-f", container_name_ams]
    subprocess.run(cmd)

    cmd = ["docker", "image", "rm", "-f", "ams:latest"]
    subprocess.run(cmd)

    # remove adjusted script file
    cmd = ["rm", "-rf", AMS_START_SCRIPT_PATH_OFFCIAL]
    subprocess.run(cmd)

    # remove adjusted config file
    cmd = ["rm", "-rf", CONF_PATH_OFFCIAL]
    subprocess.run(cmd)

    # remove tmp config file
    cmd = ["rm", "-rf", CONFIG_PATH_TEMP]
    subprocess.run(cmd)


def run_ovms(param):
    # adjust file with params for ams image build
    copy_config_file_ovms(nireq=param["nireq"], plugin_config=param["plugin_config"])

    # run ovms container
    container_name_ovms = "ovms_{}_cores".format(param["cores"])
    cmd = ["docker", "run", "--cpus={}".format(param["cores"]),
           "-v", "{}:/opt/models:ro".format(MODEL_PATH_FOR_OVMS), "--name", container_name_ovms,
           "-d", "-p", "{}:{}".format(OVMS_PORT, OVMS_PORT), "ie-serving-py:latest", "/ie-serving-py/start_server.sh",
           "ie_serving", "config", "--config_path", CONFIG_PATH_INTERNAL,
           "--port", OVMS_PORT, "--grpc_workers", param["grpc_workers"]]
    subprocess.run(cmd)
    time.sleep(10)

    return container_name_ovms


def cleanup_ovms(container_name_ovms):
    cmd = ["docker", "rm", "-f", container_name_ovms]
    subprocess.run(cmd)

    # remove ovms config file
    cmd = ["rm", "-rf", os.path.join(MODEL_PATH_FOR_OVMS, "ovms_config.json")]
    subprocess.run(cmd)

    # remove adjusted config file
    cmd = ["rm", "-rf", CONFIG_PATH_TEMP]
    subprocess.run(cmd)


def prepare_dataset_for_ovms(image):
    cmd = ["mkdir", OVMS_DATASET]
    subprocess.run(cmd)

    cmd = ["cp", os.path.join(DATASET, image),
           os.path.join(OVMS_DATASET, image)]
    subprocess.run(cmd)


def cleanup_dataset():
    cmd = ["rm", "-rf", OVMS_DATASET]
    subprocess.run(cmd)
