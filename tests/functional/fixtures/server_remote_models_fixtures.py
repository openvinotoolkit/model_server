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
from utils.model_management import (wait_endpoint_setup, minio_condition)
from botocore.client import Config


@pytest.fixture(scope="class")
def start_server_single_model_from_gc(request, get_image, get_test_dir,
                                      get_docker_context):
    client = get_docker_context
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path " \
              "gs://public-artifacts/intelai_public_models/resnet_50_i8/ " \
              "--port 9006 --target_device CPU --nireq 4 --plugin_config " \
              "\"{\\\"CPU_THROUGHPUT_STREAMS\\\": \\\"2\\\", " \
              "\\\"CPU_THREADS_NUM\\\": \\\"4\\\"}\""
    envs = ['https_proxy=' + os.getenv('https_proxy', "")]
    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-gs',
                                      ports={'9006/tcp': 9006},
                                      remove=True,
                                      environment=envs,
                                      command=command)

    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_single_model_from_s3(request, get_image, get_test_dir,
                                      get_docker_context):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    client = get_docker_context
    envs = ['AWS_ACCESS_KEY_ID=' + AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_REGION=' + AWS_REGION]
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path s3://inference-test-aipg/resnet_v1_50 " \
              "--port 9000"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-s3',
                                      ports={'9002/tcp': 9002},
                                      remove=True,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="session")
def get_docker_network(request, get_docker_context):

    client = get_docker_context
    existing = None

    try:
        existing = client.networks.get("minio-network")
    except Exception as e:
        pass

    if existing is not None:
        existing.remove()

    network = client.networks.create("minio-network")

    request.addfinalizer(network.remove)

    return network


@pytest.fixture(scope="session")
def start_minio_server(request, get_image, get_test_dir, get_docker_network,
                       get_docker_context):

    """sudo docker run -d -p 9099:9000 minio/minio server /data"""
    client = get_docker_context
    envs = []
    command = "server /data"

    client.images.pull('minio/minio:latest')

    network = get_docker_network

    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

    if MINIO_ACCESS_KEY is None or MINIO_SECRET_KEY is None:
        MINIO_ACCESS_KEY = "MINIO_A_KEY"
        MINIO_SECRET_KEY = "MINIO_S_KEY"
        os.environ["MINIO_ACCESS_KEY"] = "MINIO_A_KEY"
        os.environ["MINIO_SECRET_KEY"] = "MINIO_S_KEY"

    envs = ['MINIO_ACCESS_KEY=' + MINIO_ACCESS_KEY,
            'MINIO_SECRET_KEY=' + MINIO_SECRET_KEY]

    container = client.containers.run(image='minio/minio:latest', detach=True,
                                      name='minio.locals3.com',
                                      ports={'9000/tcp': 9000},
                                      remove=True,
                                      environment=envs,
                                      command=command,
                                      network=network.name)

    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container, minio_condition, 30)
    assert running is True, "minio container was not started successfully"

    return container


@pytest.fixture(scope="session")
def get_minio_server_s3(request, get_image, get_test_dir, start_minio_server):

    path_to_mount = get_test_dir + '/saved_models/resnet_V1_50/1'
    input_bin = os.path.join(path_to_mount, 'resnet_V1_50.bin')
    input_xml = os.path.join(path_to_mount, 'resnet_V1_50.xml')

    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    if AWS_REGION is None:
        AWS_REGION = "eu-central-1"
        os.environ["AWS_REGION"] = AWS_REGION

    if MINIO_ACCESS_KEY is None or MINIO_SECRET_KEY is None:
        MINIO_ACCESS_KEY = "MINIO_A_KEY"
        MINIO_SECRET_KEY = "MINIO_S_KEY"
        os.environ["MINIO_ACCESS_KEY"] = MINIO_ACCESS_KEY
        os.environ["MINIO_SECRET_KEY"] = MINIO_SECRET_KEY

    s3 = boto3.resource('s3',
                        endpoint_url='http://localhost:9000',
                        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
                        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
                        config=Config(signature_version='s3v4'),
                        region_name=AWS_REGION)

    bucket_conf = {'LocationConstraint': AWS_REGION}

    s3.create_bucket(Bucket='inference',
                     CreateBucketConfiguration=bucket_conf)

    s3.Bucket('inference').upload_file(input_bin,
                                       'resnet_v1_50/1/resnet_V1_50.bin')
    s3.Bucket('inference').upload_file(input_xml,
                                       'resnet_v1_50/1/resnet_V1_50.xml')

    return s3


@pytest.fixture(scope="class")
def start_server_single_model_from_minio(request, get_docker_network,
                                         get_minio_server_s3, get_image,
                                         get_test_dir, get_docker_context):

    network = get_docker_network

    AWS_ACCESS_KEY_ID = os.getenv('MINIO_ACCESS_KEY')
    AWS_SECRET_ACCESS_KEY = os.getenv('MINIO_SECRET_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    envs = ['MINIO_ACCESS_KEY' + AWS_ACCESS_KEY_ID,
            'MINIO_SECRET_KEY' + AWS_SECRET_ACCESS_KEY,
            'AWS_ACCESS_KEY_ID=' + AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_REGION=' + AWS_REGION,
            'S3_ENDPOINT=' + 'http://minio.locals3.com:9000']

    client = get_docker_context
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path s3://inference/resnet_v1_50 " \
              "--port 9099"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-minio',
                                      ports={'9099/tcp': 9099},
                                      remove=True,
                                      environment=envs,
                                      command=command,
                                      network=network.name)

    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)

    assert running is True, "docker container was not started successfully"

    return container
