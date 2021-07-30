#
# Copyright (c) 2019 Intel Corporation
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

import pytest

import config
from model.models_information import FaceDetection
from object_model.server import Server


@pytest.fixture(scope="class")
def start_server_face_detection_model_auto_shape(request):

    start_server_command_args = {"model_name": FaceDetection.name,
                                 "model_path": FaceDetection.model_path,
                                 "shape": "auto",
                                 "grpc_workers": 4,
                                 "nireq": 4}
    container_name_infix = "test-auto-shape"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="class")
def start_server_face_detection_model_named_shape(request):

    start_server_command_args = {"model_name": FaceDetection.name,
                                 "model_path": FaceDetection.model_path,
                                 "shape": "\"{\\\"data\\\": \\\"(1, 3, 600, 600)\\\"}\"",
                                 "grpc_workers": 4,
                                 "rest_workers": 2,
                                 "nireq": 2}
    container_name_infix = "test-named-shape"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()


@pytest.fixture(scope="class")
def start_server_face_detection_model_nonamed_shape(request):

    start_server_command_args = {"model_name": FaceDetection.name,
                                 "model_path": FaceDetection.model_path,
                                 "shape": "\"(1, 3, 600, 600)\"",
                                 "rest_workers": 4,
                                 "nireq": 2}
    container_name_infix = "test-nonamed-shape"
    server = Server(request, start_server_command_args,
                    container_name_infix, config.start_container_command,
                    target_device=config.target_device)
    return server.start()
