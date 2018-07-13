import pytest
import os
import requests
import numpy as np
import subprocess
import shutil
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.contrib.util import make_ndarray
import sys
sys.path.append(".")
from ie_serving.tensorflow_serving_api import predict_pb2 # noqa


def pytest_addoption(parser):
    parser.addoption(
        "--image", action="store", default="ie-serving-py:latest",
        help="docker image name which should be used to run tests"
    )
    parser.addoption(
        "--test_dir", action="store", default="/tmp/test_models",
        help="location where models and test data should be downloaded"
    )


@pytest.fixture
def get_image(request):
    return request.config.getoption("--image")


@pytest.fixture
def get_test_dir(request):
    return request.config.getoption("--test_dir")


def download_model(model_name, model_folder, model_version_folder, dir):
    model_url_base = "https://storage.googleapis.com/inference-eu/models_zoo/"\
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
    return dir + model_folder + model_version_folder + model_name + '.bin',\
        dir + model_folder + model_version_folder + model_name + '.xml'


@pytest.fixture(autouse=True)
def resnet_v1_50_model_downloader(get_test_dir):
    return download_model('resnet_V1_50', 'resnet_V1_50/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True)
def pnasnet_large_model_downloader(get_test_dir):
    return download_model('pnasnet_large', 'pnasnet_large/', '1/',
                          get_test_dir + '/saved_models/')


@pytest.fixture(autouse=True)
def download_two_models(get_test_dir):
    model1_info = download_model('resnet_V1_50', 'resnet_V1_50/', '1/',
                                 get_test_dir + '/saved_models/')
    model2_info = download_model('pnasnet_large', 'pnasnet_large/', '1/',
                                 get_test_dir + '/saved_models/')
    return [model1_info, model2_info]


@pytest.fixture(autouse=True)
def download_two_model_versions(get_test_dir):
    model1_info = download_model('resnet_V1_50', 'resnet/', '1/',
                                 get_test_dir+'/saved_models/')
    model2_info = download_model('resnet_V2_50', 'resnet/', '2/',
                                 get_test_dir+'/saved_models/')
    return [model1_info, model2_info]


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


@pytest.fixture(autouse=True)
def input_data_downloader_v1_224(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/resnet_V1_50/datasets/10_v1_imgs.npy', # noqa
        get_test_dir)


@pytest.fixture(autouse=True)
def input_data_downloader_v3_331(get_test_dir):
    return input_data_downloader(
        'https://storage.googleapis.com/inference-eu/models_zoo/pnasnet_large/datasets/10_331_v3_imgs.npy', # noqa
        get_test_dir)


@pytest.fixture(autouse=True)
def start_server_single_model(request, get_image, get_test_dir):
    CYAN_COLOR = '\033[36m'
    END_COLOR = '\033[0m'
    cmd = ['docker',
           'run',
           '--rm',
           '-d',
           '--name', 'ie-serving-py-test-single',
           '-v', '{}:/opt/ml:ro'.format(get_test_dir+'/saved_models/'),
           '-p', '9000:9000',
           get_image, '/ie-serving-py/start_server.sh', 'ie_serving',
           'model',
           '--model_name', 'resnet',
           '--model_path', '/opt/ml/resnet_V1_50',
           '--port', '9000'
           ]
    print('executing docker command:, {}{}{}'.format(CYAN_COLOR, ' '.join(cmd),
                                                     END_COLOR))

    def stop_docker():

        print("stopping docker container...")
        return_code = subprocess.call(
            "for I in `docker ps -f 'name=ie-serving-py-test-single' -q` ;"
            "do echo $I; docker stop $I; done",
            shell=True)
        if return_code == 0:
            print("docker container removed")
    request.addfinalizer(stop_docker)

    return subprocess.check_call(cmd)


@pytest.fixture(autouse=True)
def start_server_multi_model(request, get_image, get_test_dir):

    shutil.copyfile('tests/functional/config.json',
                    get_test_dir + '/saved_models/config.json')

    CYAN_COLOR = '\033[36m'
    END_COLOR = '\033[0m'
    cmd = ['docker',
           'run',
           '--rm',
           '-d',
           '--name', 'ie-serving-py-test-multi',
           '-v', '{}:/opt/ml:ro'.format(get_test_dir+'/saved_models/'),
           '-p', '9001:9001',
           get_image, '/ie-serving-py/start_server.sh', 'ie_serving',
           'config',
           '--config_path', '/opt/ml/config.json',
           '--port', '9001'
           ]
    print('executing docker command:, {}{}{}'.format(CYAN_COLOR, ' '.join(cmd),
                                                     END_COLOR))

    def stop_docker():

        print("stopping docker container...")
        return_code = subprocess.call(
            "for I in `docker ps -f 'name=ie-serving-py-test-multi' -q` ;"
            "do echo $I; docker stop $I; done",
            shell=True)
        if return_code == 0:
            print("docker container removed")
    request.addfinalizer(stop_docker)

    return subprocess.check_call(cmd)


def infer(imgs, slice_number, input_tensor, grpc_stub, model_spec_name,
          model_spec_version, output_tensor):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    if model_spec_version is not None:
        request.model_spec.version.value = model_spec_version
    img = imgs[slice_number, ...]
    print("input shape", list((1,) + img.shape))
    request.inputs[input_tensor].CopyFrom(
        make_tensor_proto(img, shape=list((1,) + img.shape)))
    result = grpc_stub.Predict(request, 10.0)

    return make_ndarray(result.outputs[output_tensor])
