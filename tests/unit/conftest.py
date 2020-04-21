#
# Copyright (c) 2018-2020 Intel Corporation
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
import grpc_testing
import numpy as np
import pytest
import queue
from config import DEFAULT_INPUT_KEY, DEFAULT_OUTPUT_KEY
from falcon import testing
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, \
    get_model_status_pb2, model_service_pb2

from ie_serving.models.ir_engine import IrEngine
from ie_serving.models.local_model import LocalModel
from ie_serving.models.model_version_status import ModelVersionStatus
from ie_serving.models.shape_management.batching_info import BatchingInfo
from ie_serving.models.shape_management.shape_info import ShapeInfo
from ie_serving.server.rest_service import create_rest_api
from ie_serving.server.service import PredictionServiceServicer, \
    ModelServiceServicer

PREDICT_SERVICE = prediction_service_pb2. \
    DESCRIPTOR.services_by_name['PredictionService']

MODEL_SERVICE = model_service_pb2. \
    DESCRIPTOR.services_by_name['ModelService']


class MockedIOInfo:
    def __init__(self, precision, shape, layout):
        self.precision = precision
        self.shape = shape
        self.layout = layout


class MockedNet:
    def __init__(self, inputs: dict, outputs: dict):
        self.inputs = inputs
        self.outputs = outputs


class MockedExecNet:

    class MockerInferRequest:
        def __init__(self):
            self.outputs = {}

        def set_completion_callback(self, py_callback, py_data):
            pass

        def async_infer(self, inference_input):
            pass

    def __init__(self):
        self.requests = [self.MockerInferRequest(), self.MockerInferRequest()]


@pytest.fixture
def get_fake_model():
    mapping_config = 'mapping_config.json'
    exec_net = MockedExecNet()
    net = MockedNet(
        inputs={DEFAULT_INPUT_KEY: MockedIOInfo('FP32', [1, 1, 1], 'NCHW')},
        outputs={DEFAULT_OUTPUT_KEY: MockedIOInfo('FP32', [1, 1, 1], 'NCHW')})
    plugin = None
    batching_info = BatchingInfo(None)
    shape_info = ShapeInfo(None, net.inputs)
    new_engines = {}
    available_versions = [1, 2, 3]
    requests_queue = queue.Queue()
    free_ireq_index_queue = queue.Queue(maxsize=1)
    free_ireq_index_queue.put(0)
    for version in available_versions:
        engine = IrEngine(model_name='test', model_version=version,
                          mapping_config=mapping_config, exec_net=exec_net,
                          net=net, plugin=plugin, batching_info=batching_info,
                          shape_info=shape_info, target_device="CPU",
                          free_ireq_index_queue=free_ireq_index_queue,
                          plugin_config=None, num_ireq=1,
                          requests_queue=requests_queue)
        new_engines.update({version: engine})
    model_name = "test"
    versions_statuses = {}
    batch_size_param, shape_param = None, None
    for version in available_versions:
        versions_statuses[version] = ModelVersionStatus(model_name, version)
    new_model = LocalModel(model_name=model_name,
                           model_directory='fake_path/model/',
                           available_versions=available_versions,
                           engines=new_engines,
                           batch_size_param=batch_size_param,
                           shape_param=shape_param,
                           version_policy_filter=lambda versions: versions[:],
                           versions_statuses=versions_statuses,
                           update_locks={},
                           plugin_config=None, target_device="CPU",
                           num_ireq=1)
    return new_model


@pytest.fixture
def get_fake_ir_engine():
    mapping_config = 'mapping_config.json'
    exec_net = MockedExecNet()
    net = MockedNet(
        inputs={DEFAULT_INPUT_KEY: MockedIOInfo('FP32', [1, 1, 1], 'NCHW')},
        outputs={DEFAULT_OUTPUT_KEY: MockedIOInfo('FP32', [1, 1, 1], 'NCHW')})
    plugin = None
    batching_info = BatchingInfo(None)
    shape_info = ShapeInfo(None, net.inputs)
    requests_queue = queue.Queue()
    free_ireq_index_queue = queue.Queue(maxsize=1)
    free_ireq_index_queue.put(0)
    engine = IrEngine(model_name='test', model_version=1,
                      mapping_config=mapping_config, exec_net=exec_net,
                      net=net, plugin=plugin, batching_info=batching_info,
                      shape_info=shape_info, target_device="CPU",
                      free_ireq_index_queue=free_ireq_index_queue,
                      plugin_config=None, num_ireq=1,
                      requests_queue=requests_queue)
    return engine


@pytest.fixture
def get_grpc_service_for_predict(get_fake_model):
    _real_time = grpc_testing.strict_real_time()
    servicer = PredictionServiceServicer(models={'test': get_fake_model})
    descriptors_to_servicers = {
        PREDICT_SERVICE: servicer
    }
    _real_time_server = grpc_testing.server_from_dictionary(
        descriptors_to_servicers, _real_time)

    return _real_time_server


@pytest.fixture
def get_grpc_service_for_model_status(get_fake_model):
    _real_time = grpc_testing.strict_real_time()
    servicer = ModelServiceServicer(models={'test': get_fake_model})
    descriptors_to_servicers = {
        MODEL_SERVICE: servicer
    }
    _real_time_server = grpc_testing.server_from_dictionary(
        descriptors_to_servicers, _real_time)

    return _real_time_server


def get_fake_request(model_name, data_shape, input_blob, version=None):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    if version is not None:
        request.model_spec.version.value = version
    data = np.ones(shape=data_shape)
    request.inputs[input_blob].CopyFrom(
        make_tensor_proto(data, shape=data.shape))
    return request


def get_fake_model_metadata_request(model_name, metadata_field, version=None):
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = model_name
    if version is not None:
        request.model_spec.version.value = version
    request.metadata_field.append(metadata_field)
    return request


def get_fake_model_status_request(model_name, version=None):
    request = get_model_status_pb2.GetModelStatusRequest()
    request.model_spec.name = model_name
    if version is not None:
        request.model_spec.version.value = version
    return request


@pytest.fixture()
def client(get_fake_model):
    rest_api = create_rest_api(models={"test": get_fake_model})
    return testing.TestClient(rest_api)
