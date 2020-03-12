#
# Copyright (c) 2018-2019 Intel Corporation
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

import docker
import grpc
import pytest
from constants import MODEL_SERVICE, PREDICTION_SERVICE
from tensorflow_serving.apis import prediction_service_pb2_grpc, \
    model_service_pb2_grpc  # noqa

pytest_plugins = [
    'fixtures.model_download_fixtures',
    'fixtures.model_conversion_fixtures',
    'fixtures.server_detection_model_fixtures',
    'fixtures.server_for_update_fixtures',
    'fixtures.server_local_models_fixtures',
    'fixtures.server_multi_model_fixtures',
    'fixtures.server_remote_models_fixtures',
    'fixtures.server_with_batching_fixtures',
    'fixtures.server_with_version_policy_fixtures',
    ]


def pytest_addoption(parser):
    parser.addoption(
        "--image", action="store", default="ie-serving-py:latest",
        help="docker image name which should be used to run tests"
    )
    parser.addoption(
        "--test_dir", action="store", default="/tmp/ovms_models",
        help="location where models and test data should be downloaded"
    )


@pytest.fixture(scope="session")
def get_image(request):
    return request.config.getoption("--image")


@pytest.fixture(scope="session")
def get_test_dir(request):
    os.makedirs(request.config.getoption("--test_dir"), exist_ok=True)
    return request.config.getoption("--test_dir")


@pytest.fixture(scope="session")
def get_docker_context():
    return docker.from_env()


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
