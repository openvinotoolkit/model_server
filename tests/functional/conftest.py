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

import docker
import numpy as np
import json
import os
import pytest
import requests
import shutil
import sys
import time
from distutils.dir_util import copy_tree
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.contrib.util import make_ndarray
from grpc.beta import implementations

sys.path.append(".")
from ie_serving.tensorflow_serving_api import predict_pb2, \
    get_model_metadata_pb2, prediction_service_pb2  # noqa


ERROR_SHAPE = 'response has invalid output'


def pytest_addoption(parser):
    parser.addoption(
        "--image", action="store", default="ie-serving-py:latest",
        help="docker image name which should be used to run tests"
    )
    parser.addoption(
        "--test_dir", action="store", default="/tmp/test_models",
        help="location where models and test data should be downloaded"
    )


@pytest.fixture(scope="session")
def get_image(request):
    return request.config.getoption("--image")


@pytest.fixture(scope="session")
def get_test_dir(request):
    return request.config.getoption("--test_dir")


@pytest.fixture(scope="session")
def get_docker_context():
    return docker.from_env()


def download_model(model_name, model_folder, model_version_folder, dir):
    model_url_base = "https://storage.googleapis.com/inference-eu/models_zoo/" \
                     + model_name + "/frozen_" + model_name

    if not os.path.exists(dir + model_folder + model_version_folder):
        print("Downloading "+model_name+" model...")
        print(dir)
        os.makedirs(dir + model_folder + model_version_folder)
        response = requests.get(model_url_base + '.bin', stream=True)
        with open(
                dir + model_folder + model_version_folder + model_name + '.bin',
                'wb') as output:
            output.write(response.content)
        response = requests.get(model_url_base + '.xml', stream=True)
        with open(
                dir + model_folder + model_version_folder + model_name + '.xml',
                'wb') as output:
            output.write(response.content)
    return dir + model_folder + model_version_folder + model_name + '.bin', \
           dir + model_folder + model_version_folder + model_name + '.xml'


@pytest.fixture(autouse=True, scope="session")
def resnet_v1_50_model_downloader(get_test_dir):
    return download_model('resnet_V1_50', 'resnet_V1_50/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def pnasnet_large_model_downloader(get_test_dir):
    return download_model('pnasnet_large', 'pnasnet_large/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def resnet_2_out_model_downloader(get_test_dir):
    return download_model('resnet_2_out', 'resnet_2_out/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def resnet_8_batch_model_downloader(get_test_dir):
    return download_model('resnet_V1_50_batch8', 'resnet_V1_50_batch8/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True, scope="session")
def download_two_models(get_test_dir):
    model1_info = download_model('resnet_V1_50', 'resnet_V1_50/', '1/',
                                 get_test_dir + '/saved_models/')
    model2_info = download_model('pnasnet_large', 'pnasnet_large/', '1/',
                                 get_test_dir + '/saved_models/')
    return [model1_info, model2_info]


@pytest.fixture(autouse=True, scope="session")
def download_two_model_versions(get_test_dir):
    model1_info = download_model('resnet_V1_50', 'resnet/', '1/',
                                 get_test_dir+'/saved_models/')
    model2_info = download_model('resnet_V2_50', 'resnet/', '2/',
                                 get_test_dir+'/saved_models/')
    return [model1_info, model2_info]


def copy_model(model, version, destination_path):
    dir_to_cpy = destination_path + str(version)
    if not os.path.exists(dir_to_cpy):
        os.makedirs(dir_to_cpy)
        shutil.copy(model[0], dir_to_cpy + '/model.bin')
        shutil.copy(model[1], dir_to_cpy + '/model.xml')
    return dir_to_cpy


@pytest.fixture(autouse=True, scope="session")
def model_version_policy_models(get_test_dir,
                                download_two_model_versions,
                                resnet_2_out_model_downloader):
    model_ver_dir = os.path.join(get_test_dir, 'saved_models', 'model_ver')
    resnets = download_two_model_versions
    resnet_1 = os.path.dirname(resnets[0][0])
    resnet_1_dir = os.path.join(model_ver_dir, '1')
    resnet_2 = os.path.dirname(resnets[1][0])
    resnet_2_dir = os.path.join(model_ver_dir, '2')
    resnet_2_out = os.path.dirname(resnet_2_out_model_downloader[0])
    resnet_2_out_dir = os.path.join(model_ver_dir, '3')
    if not os.path.exists(model_ver_dir):
        os.makedirs(model_ver_dir)
        copy_tree(resnet_1, resnet_1_dir)
        copy_tree(resnet_2, resnet_2_dir)
        copy_tree(resnet_2_out, resnet_2_out_dir)

    return resnet_1_dir, resnet_2_dir, resnet_2_out_dir


def input_data_downloader(numpy_url, get_test_dir):
    filename = numpy_url.split("/")[-1]
    if not os.path.exists(get_test_dir + '/' + filename):
        response = requests.get(numpy_url, stream=True)
        with open(get_test_dir + '/' + filename, 'wb') as output:
            output.write(response.content)
    imgs = np.load(get_test_dir + '/' + filename, mmap_mode='r',
                   allow_pickle=False)
    imgs = imgs.transpose((0, 3, 1, 2))  # transpose to adjust from NHWC>NCHW
    print(imgs.shape)
    return imgs


@pytest.fixture(autouse=True, scope="session")
def input_data_downloader_v1_224(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/resnet_V1_50/datasets/10_v1_imgs.npy', # noqa
        get_test_dir)


@pytest.fixture(autouse=True, scope="session")
def input_data_downloader_v3_331(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/pnasnet_large/datasets/10_331_v3_imgs.npy', # noqa
        get_test_dir)


@pytest.fixture(autouse=True, scope="session")
def create_channel_for_port_multi_server():
    channel = implementations.insecure_channel('localhost', 9001)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(autouse=True, scope="session")
def create_channel_for_port_single_server():
    channel = implementations.insecure_channel('localhost', 9000)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="session")
def create_channel_for_port_mapping_server():
    channel = implementations.insecure_channel('localhost', 9002)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="session")
def create_channel_for_batching_server():
    channel = implementations.insecure_channel('localhost', 9003)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="session")
def create_channel_for_batching_server_bs4():
    channel = implementations.insecure_channel('localhost', 9004)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="session")
def create_channel_for_batching_server_auto():
    channel = implementations.insecure_channel('localhost', 9005)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="session")
def create_channel_for_model_ver_pol_server():
    channel = implementations.insecure_channel('localhost', 9006)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="session")
def create_channel_for_update_flow_latest():
    channel = implementations.insecure_channel('localhost', 9007)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="session")
def create_channel_for_update_flow_specific():
    channel = implementations.insecure_channel('localhost', 9008)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


@pytest.fixture(scope="class")
def start_server_single_model(request, get_image, get_test_dir,
                              get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir+'/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50 " \
              "--port 9000 --rest-port 5555"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single',
                                      ports={'9000/tcp': 9000,
                                             '5555/tcp': 5555},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_single_model_from_gc(request, get_image, get_test_dir,
                                      get_docker_context):
    GOOGLE_APPLICATION_CREDENTIALS = \
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    client = get_docker_context
    envs = ['GOOGLE_APPLICATION_CREDENTIALS=/etc/gcp.json']
    volumes_dict = {GOOGLE_APPLICATION_CREDENTIALS: {'bind': '/etc/gcp.json',
                                                     'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet " \
              "--model_path gs://inference-eu/ml-test " \
              "--port 9000"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-single-gs',
                                      ports={'9000/tcp': 9000},
                                      remove=True, volumes=volumes_dict,
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
                                      ports={'9000/tcp': 9000},
                                      remove=True,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_with_mapping(request, get_image, get_test_dir,
                              get_docker_context):

    shutil.copyfile('tests/functional/mapping_config.json',
                    get_test_dir + '/saved_models/resnet_2_out/1/'
                                   'mapping_config.json')
    client = get_docker_context
    path_to_mount = get_test_dir+'/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet_2_out --model_path /opt/ml/resnet_2_out " \
              "--port 9002 --rest-port 5556"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-2-out',
                                      ports={'9002/tcp': 9002,
                                             '5556/tcp': 5556},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="session")
def start_server_multi_model(request, get_image, get_test_dir,
                             get_docker_context):

    shutil.copyfile('tests/functional/config.json',
                    get_test_dir + '/saved_models/config.json')

    GOOGLE_APPLICATION_CREDENTIALS = \
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    client = get_docker_context
    envs = ['GOOGLE_APPLICATION_CREDENTIALS=/etc/gcp.json',
            'AWS_ACCESS_KEY_ID=' + AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY=' + AWS_SECRET_ACCESS_KEY,
            'AWS_REGION=' + AWS_REGION]
    volumes_dict = {'{}'.format(get_test_dir+'/saved_models/'):
                        {'bind': '/opt/ml', 'mode': 'ro'},
                    GOOGLE_APPLICATION_CREDENTIALS:
                        {'bind': '/etc/gcp.json', 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving config " \
              "--config_path /opt/ml/config.json --port 9001"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-multi',
                                      ports={'9001/tcp': 9001},
                                      remove=True, volumes=volumes_dict,
                                      environment=envs,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model(request, get_image, get_test_dir,
                             get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir+'/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9003"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch',
                                      ports={'9003/tcp': 9003},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_auto(request, get_image, get_test_dir,
                                  get_docker_context):

    client = get_docker_context
    path_to_mount = get_test_dir+'/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9005 --batch_size auto"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-autobatch',
                                      ports={'9005/tcp': 9005},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_batch_model_bs4(request, get_image, get_test_dir,
                                 get_docker_context):

    client = get_docker_context
    path_to_mount = get_test_dir+'/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/resnet_V1_50_batch8 " \
              "--port 9004 --batch_size 4"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-batch4',
                                      ports={'9004/tcp': 9004},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_model_ver_policy(request, get_image, get_test_dir,
                                  get_docker_context):

    shutil.copyfile('tests/functional/model_version_policy_config.json',
                    get_test_dir +
                    '/saved_models/model_ver_policy_config.json')

    shutil.copyfile('tests/functional/mapping_config.json',
                    get_test_dir + '/saved_models/model_ver/3/'
                                   'mapping_config.json')

    client = get_docker_context
    volumes_dict = {'{}'.format(get_test_dir+'/saved_models/'):
                        {'bind': '/opt/ml', 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving config " \
              "--config_path /opt/ml/model_ver_policy_config.json --port 9006"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-policy',
                                      ports={'9006/tcp': 9006},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_update_flow_latest(request, get_image, get_test_dir,
                                    get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir+'/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = "/ie-serving-py/start_server.sh ie_serving model " \
              "--model_name resnet --model_path /opt/ml/update " \
              "--port 9007"

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-update-latest',
                                      ports={'9007/tcp': 9007},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


@pytest.fixture(scope="class")
def start_server_update_flow_specific(request, get_image, get_test_dir,
                                      get_docker_context):
    client = get_docker_context
    path_to_mount = get_test_dir+'/saved_models/'
    volumes_dict = {'{}'.format(path_to_mount): {'bind': '/opt/ml',
                                                 'mode': 'ro'}}
    command = '/ie-serving-py/start_server.sh ie_serving model ' \
              '--model_name resnet --model_path /opt/ml/update ' \
              '--port 9008 --model_version_policy' \
              ' \'{"specific": { "versions":[1, 3, 4] }}\' '

    container = client.containers.run(image=get_image, detach=True,
                                      name='ie-serving-py-test-'
                                           'update-specific',
                                      ports={'9008/tcp': 9008},
                                      remove=True, volumes=volumes_dict,
                                      command=command)
    request.addfinalizer(container.kill)

    running = wait_endpoint_setup(container)
    assert running is True, "docker container was not started successfully"

    return container


def wait_endpoint_setup(container):
    start_time = time.time()
    tick = start_time
    running = False
    logs = ""
    while tick - start_time < 300:
        tick = time.time()
        try:
            logs = str(container.logs())
            if "Server listens on port" in logs:
                running = True
                break
        except Exception as e:
            time.sleep(1)
    print("Logs from container: ", logs)
    return running


def infer(imgs, slice_number, input_tensor, grpc_stub, model_spec_name,
          model_spec_version, output_tensors):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    if model_spec_version is not None:
        request.model_spec.version.value = model_spec_version
    img = imgs[slice_number, ...]
    print("input shape", list((1,) + img.shape))
    request.inputs[input_tensor].CopyFrom(
        make_tensor_proto(img, shape=list((1,) + img.shape)))
    result = grpc_stub.Predict(request, 10.0)
    data = {}
    for output_tensor in output_tensors:
        data[output_tensor] = make_ndarray(result.outputs[output_tensor])
    return data


def infer_batch(batch_input, input_tensor, grpc_stub, model_spec_name,
                model_spec_version, output_tensors):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    if model_spec_version is not None:
        request.model_spec.version.value = model_spec_version
    print("input shape", list(batch_input.shape))
    request.inputs[input_tensor].CopyFrom(
        make_tensor_proto(batch_input, shape=list(batch_input.shape)))
    result = grpc_stub.Predict(request, 10.0)
    data = {}
    for output_tensor in output_tensors:
        data[output_tensor] = make_ndarray(result.outputs[output_tensor])
    return data


def prepare_body_format(img, request_format, input_name):
    signature = "serving_default"
    if request_format == "row_name":
        instances = []
        for i in range(0, img.shape[0], 1):
            instances.append({input_name: img[i].tolist()})
        data_obj = {"signature_name": signature, "instances": instances}
    elif request_format == "row_noname":
        data_obj = {"signature_name": signature, 'instances': img.tolist()}
    elif request_format == "column_name":
        data_obj = {"signature_name": signature,
                    'inputs': {input_name: img.tolist()}}
    elif request_format == "column_noname":
        data_obj = {"signature_name": signature, 'inputs':  img.tolist()}
    data_json = json.dumps(data_obj)
    return data_json


def process_json_output(result_dict, output_tensors):
    output = {}
    if "outputs" in result_dict:
        keyname = "outputs"
        if type(result_dict[keyname]) is dict:
            for output_tensor in output_tensors:
                output[output_tensor] = np.asarray(result_dict[keyname][output_tensor])
        else:
            output[output_tensors[0]] = np.asarray(result_dict[keyname])
    elif "predictions" in result_dict:
        keyname = "predictions"
        if type(result_dict[keyname][0]) is dict:
            for row in result_dict[keyname]:
                print(row.keys())
                for output_tensor in output_tensors:
                    if output_tensor not in output:
                        output[output_tensor] = []
                    output[output_tensor].append(row[output_tensor])
            for output_tensor in output_tensors:
                output[output_tensor] = np.asarray(output[output_tensor])
        else:
            output[output_tensors[0]] = np.asarray(result_dict[keyname])
    else:
        print("Missing required response in {}".format(result_dict))

    return output


def infer_rest(imgs, slice_number, input_tensor, rest_url, model_spec_name,
               model_spec_version, output_tensors, request_format):
    signature = "serving_default"
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    if model_spec_version is not None:
        request.model_spec.version.value = model_spec_version
    img = imgs[slice_number:slice_number + 1]
    print("input shape", img.shape)
    data_json = prepare_body_format(img, request_format, input_tensor)
    result = requests.post(rest_url, data=data_json)
    output_json = json.loads(result.text)
    data = process_json_output(output_json, output_tensors)
    return data


def get_model_metadata(model_name, metadata_field: str="signature_def",
                       version=None):
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = model_name
    if version is not None:
        request.model_spec.version.value = version
    request.metadata_field.append(metadata_field)
    return request


def model_metadata_response(response):
    signature_def = response.metadata['signature_def']
    signature_map = get_model_metadata_pb2.SignatureDefMap()
    signature_map.ParseFromString(signature_def.value)
    serving_default = signature_map.ListFields()[0][1]['serving_default']
    serving_inputs = serving_default.inputs
    input_blobs_keys = {key: {} for key in serving_inputs.keys()}
    tensor_shape = {key: serving_inputs[key].tensor_shape
                    for key in serving_inputs.keys()}
    for input_blob in input_blobs_keys:
        inputs_shape = [d.size for d in tensor_shape[input_blob].dim]
        tensor_dtype = serving_inputs[input_blob].dtype
        input_blobs_keys[input_blob].update({'shape': inputs_shape})
        input_blobs_keys[input_blob].update({'dtype': tensor_dtype})

    serving_outputs = serving_default.outputs
    output_blobs_keys = {key: {} for key in serving_outputs.keys()}
    tensor_shape = {key: serving_outputs[key].tensor_shape
                    for key in serving_outputs.keys()}
    for output_blob in output_blobs_keys:
        outputs_shape = [d.size for d in tensor_shape[output_blob].dim]
        tensor_dtype = serving_outputs[output_blob].dtype
        output_blobs_keys[output_blob].update({'shape': outputs_shape})
        output_blobs_keys[output_blob].update({'dtype': tensor_dtype})

    return input_blobs_keys, output_blobs_keys
