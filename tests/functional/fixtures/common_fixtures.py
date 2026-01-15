#
# Copyright (c) 2026 Intel Corporation
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

import docker
import grpc  # noqa
import logging
import re

import pytest

from tensorflow_serving.apis import prediction_service_pb2_grpc, model_service_pb2_grpc  # noqa

from tests.functional.config import image
from tests.functional.constants.constants import MODEL_SERVICE, PREDICTION_SERVICE
from tests.functional.utils.cleanup import get_docker_client

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def get_docker_context(request):
    client = get_docker_client()
    request.addfinalizer(client.close)
    return client


@pytest.fixture()
def create_grpc_channel():
    def _create_channel(address: str, service: int):
        channel = grpc.insecure_channel(address)
        if service == MODEL_SERVICE:
            return model_service_pb2_grpc.ModelServiceStub(channel)
        elif service == PREDICTION_SERVICE:
            return prediction_service_pb2_grpc.PredictionServiceStub(channel)
        return None

    return _create_channel


def get_docker_image_os_version_from_container():
    client = docker.from_env()
    cmd = 'cat /etc/os-release'
    os_distname = "__invalid__"
    try:
        output = client.containers.run(image=image, entrypoint=cmd)
        output = output.decode("utf-8")
        os_distname = re.search('^PRETTY_NAME="(.+)"\n', output, re.MULTILINE).group(1)
    except AttributeError as e:
        logger.error(f"Cannot find complete os version information.\n{cmd}\n{output}")

    return os_distname


def get_ov_and_ovms_version_from_container():
    client = docker.from_env()
    cmd = "/ovms/bin/ovms --version"
    _ov_version, _ovms_version = ["__invalid__"] * 2
    try:
        output = client.containers.run(image=image, entrypoint=cmd)
        output = output.decode("utf-8")
        _ovms_version = re.search('OpenVINO Model Server (.+)\n', output, re.MULTILINE).group(1)
        _ov_version = re.search('OpenVINO backend (.+)\n', output, re.MULTILINE).group(1)
    except AttributeError as e:
        logger.error(f"Cannot find complete ovms version information.\n{cmd}\n{output}")

    return _ov_version, _ovms_version
