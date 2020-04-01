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

import docker
from docker.errors import APIError
from utils.parametrization import get_tests_suffix


def get_docker_client():
    return docker.from_env()


def clean_hanging_containers():
    client = get_docker_client()
    containers = client.containers.list(all=True)
    tests_suffix = get_tests_suffix()
    for container in containers:
        if tests_suffix in container.name:
            kill_container(container)
            remove_container(container)


def kill_container(container):
    try:
        container.kill()
    except APIError:
        pass


def remove_container(container):
    try:
        container.remove()
    except APIError:
        pass
