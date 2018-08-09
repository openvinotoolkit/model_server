import pytest
import os
import requests
import numpy as np
import subprocess
import shutil
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.contrib.util import make_ndarray
from grpc.beta import implementations
import sys
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


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
def start_server_with_mapping(request, get_image, get_test_dir):

    shutil.copyfile('tests/functional/mapping_config.json',
                    get_test_dir + '/saved_models/resnet_2_out/1/'
                                   'mapping_config.json')

    CYAN_COLOR = '\033[36m'
    END_COLOR = '\033[0m'
    cmd = ['docker',
           'run',
           '--rm',
           '-d',
           '--name', 'ie-serving-py-test-2-out',
           '-v', '{}:/opt/ml:ro'.format(get_test_dir+'/saved_models/'),
           '-p', '9002:9002',
           get_image, '/ie-serving-py/start_server.sh', 'ie_serving',
           'model',
           '--model_name', 'resnet_2_out',
           '--model_path', '/opt/ml/resnet_2_out',
           '--port', '9002'
           ]
    print('executing docker command:, {}{}{}'.format(CYAN_COLOR, ' '.join(cmd),
                                                     END_COLOR))

    def stop_docker():

        print("stopping docker container...")
        return_code = subprocess.call(
            "for I in `docker ps -f 'name=ie-serving-py-test-2-out' -q` ;"
            "do echo $I; docker stop $I; done",
            shell=True)
        if return_code == 0:
            print("docker container removed")
    request.addfinalizer(stop_docker)

    return subprocess.check_call(cmd)


@pytest.fixture(scope="session")
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
