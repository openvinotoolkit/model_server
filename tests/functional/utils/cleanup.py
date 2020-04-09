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


def clean_hanging_docker_resources():
    client = get_docker_client()
    containers = client.containers.list(all=True)
    networks = client.networks.list()
    tests_suffix = get_tests_suffix()
    clean_hanging_containers(containers, tests_suffix)
    clean_hanging_networks(networks, tests_suffix)


def clean_hanging_containers(containers, tests_suffix):
    for container in containers:
        if tests_suffix in container.name:
            kill_container(container)
            remove_resource(container)


def clean_hanging_networks(networks, tests_suffix):
    for network in networks:
        if tests_suffix in network.name:
            remove_resource(network)


def kill_container(container):
    try:
        container.kill()
    except APIError as e:
        handle_cleanup_exception(e)


def remove_resource(resource):
    try:
        resource.remove()
    except APIError as e:
        handle_cleanup_exception(e)


def handle_cleanup_exception(docker_error):
    # It is okay to have these errors as
    # it means resource not exist or being removed or killed already
    allowed_errors = [404, 409]
    if docker_error.status_code in allowed_errors:
        pass
    else:
        raise
