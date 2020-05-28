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

import os

import boto3
import pytest
from botocore.client import Config

import config
from model.models_information import Resnet, ResnetS3, ResnetGS
from utils.model_management import wait_endpoint_setup, minio_condition
from utils.parametrization import get_tests_suffix, get_ports_for_fixture
from utils.server import start_ovms_container, save_container_logs


@pytest.fixture(scope="class")
def start_server_single_model_from_gc(request, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    start_server_command_args = {"model_name": Resnet.name,
                                 "model_path": ResnetGS.model_path,
                                 "target_device": "CPU",
                                 "nireq": 4,
                                 "plugin_config": "\"{\\\"CPU_THROUGHPUT_STREAMS\\\": \\\"2\\\", "
                                                  "\\\"CPU_THREADS_NUM\\\": \\\"4\\\"}\""}
    container_name_infix = "test-single-gs"
    envs = ['https_proxy=' + os.getenv('https_proxy', "")]
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command, envs)
    return container, ports


@pytest.fixture(scope="session")
def get_docker_network(request, get_docker_context):

    client = get_docker_context
    existing = None

    try:
        existing = client.networks.get("minio-network-{}".format(
            get_tests_suffix()))
    except Exception as e:
        pass

    if existing is not None:
        existing.remove()

    network = client.networks.create("minio-network-{}".format(
        get_tests_suffix()))

    request.addfinalizer(network.remove)

    return network


@pytest.fixture(scope="session")
def start_minio_server(request, get_docker_network, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    """sudo docker run -d -p 9099:9000 minio/minio server /data"""
    client = get_docker_context

    grpc_port, rest_port = get_ports_for_fixture()

    command = 'server --address ":{}" /data'.format(grpc_port)

    client.images.pull('minio/minio:latest')

    network = get_docker_network

    minio_access_key = os.getenv('MINIO_ACCESS_KEY')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY')

    if minio_access_key is None or minio_secret_key is None:
        minio_access_key = "MINIO_A_KEY"
        minio_secret_key = "MINIO_S_KEY"
        os.environ["MINIO_ACCESS_KEY"] = "MINIO_A_KEY"
        os.environ["MINIO_SECRET_KEY"] = "MINIO_S_KEY"

    envs = ['MINIO_ACCESS_KEY=' + minio_access_key,
            'MINIO_SECRET_KEY=' + minio_secret_key]

    container = client.containers.run(image='minio/minio:latest', detach=True,
                                      name='minio.locals3-{}.com'.format(
                                          get_tests_suffix()),
                                      ports={'{}/tcp'.format(grpc_port):
                                             grpc_port},
                                      remove=True,
                                      environment=envs,
                                      command=command,
                                      network=network.name)

    running = wait_endpoint_setup(container, minio_condition, 30, "created")
    assert running is True, "minio container was not started successfully"

    return container, {"grpc_port": grpc_port, "rest_port": rest_port}


@pytest.fixture(scope="session")
def get_minio_server_s3(start_minio_server):

    path_to_mount = config.path_to_mount + '/{}/{}'.format(Resnet.name, Resnet.version)
    input_bin = os.path.join(path_to_mount, '{}.bin'.format(Resnet.name))
    input_xml = os.path.join(path_to_mount, '{}.xml'.format(Resnet.name))

    minio_access_key = os.getenv('MINIO_ACCESS_KEY')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY')
    aws_region = os.getenv('AWS_REGION')

    if aws_region is None:
        aws_region = "eu-central-1"
        os.environ["AWS_REGION"] = aws_region

    if minio_access_key is None or minio_secret_key is None:
        minio_access_key = "MINIO_A_KEY"
        minio_secret_key = "MINIO_S_KEY"
        os.environ["MINIO_ACCESS_KEY"] = minio_access_key
        os.environ["MINIO_SECRET_KEY"] = minio_secret_key

    _, ports = start_minio_server
    s3 = boto3.resource('s3',
                        endpoint_url='http://localhost:{}'.format(
                            ports["grpc_port"]),
                        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
                        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
                        config=Config(signature_version='s3v4'),
                        region_name=aws_region)

    bucket_conf = {'LocationConstraint': aws_region}

    s3.create_bucket(Bucket='inference',
                     CreateBucketConfiguration=bucket_conf)

    s3.Bucket('inference').upload_file(input_bin,
                                       '{name}/{version}/{name}.bin'.format(name=Resnet.name, version=Resnet.version))
    s3.Bucket('inference').upload_file(input_xml,
                                       '{name}/{version}/{name}.xml'.format(name=Resnet.name, version=Resnet.version))

    return s3, ports


@pytest.fixture(scope="class")
def start_server_single_model_from_minio(request, get_docker_network, get_minio_server_s3, get_docker_context):

    def finalizer():
        save_container_logs(container=container)
        container.stop()

    request.addfinalizer(finalizer)

    network = get_docker_network

    aws_access_key_id = os.getenv('MINIO_ACCESS_KEY')
    aws_secret_access_key = os.getenv('MINIO_SECRET_KEY')
    aws_region = os.getenv('AWS_REGION')

    _, ports = get_minio_server_s3
    grpc_port = ports["grpc_port"]
    minio_endpoint = 'http://minio.locals3-{}.com:{}'.format(
        get_tests_suffix(), grpc_port)

    envs = ['MINIO_ACCESS_KEY=' + aws_access_key_id,
            'MINIO_SECRET_KEY=' + aws_secret_access_key,
            'AWS_ACCESS_KEY_ID=' + aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY=' + aws_secret_access_key,
            'AWS_REGION=' + aws_region,
            'S3_ENDPOINT=' + minio_endpoint]

    start_server_command_args = {"model_name": Resnet.name,
                                 "model_path": ResnetS3.model_path}
    container_name_infix = "test-single-minio"
    container, ports = start_ovms_container(get_docker_context, start_server_command_args,
                                            container_name_infix, config.start_container_command, envs, network.name)

    return container, ports
