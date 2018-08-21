#
# Copyright (c) 2018 Intel Corporation
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

from ie_serving.tensorflow_serving_api import prediction_service_pb2
from ie_serving.tensorflow_serving_api import predict_pb2
from ie_serving.tensorflow_serving_api import get_model_metadata_pb2
from ie_serving.server.service import PredictionServiceServicer
from ie_serving.models.model import Model
from ie_serving.models.ir_engine import IrEngine
from tensorflow.contrib.util import make_tensor_proto
import grpc_testing
import numpy as np
import pytest

PREDICT_SERVICE = prediction_service_pb2.\
                  DESCRIPTOR.services_by_name['PredictionService']


@pytest.fixture
def get_fake_model():
    model_xml = 'model1.xml'
    model_bin = 'model1.bin'
    exec_net = None
    input_key = 'input'
    inputs = {input_key: [1, 1]}
    outputs = ['test_output']
    engine = IrEngine(model_bin=model_bin, model_xml=model_xml,
                      exec_net=exec_net, inputs=inputs, outputs=outputs)
    new_engines = {1: engine, 2: engine, 3: engine}
    new_model = Model(model_name="test", model_directory='fake_path/model/',
                      available_versions=[1, 2, 3], engines=new_engines)
    return new_model


@pytest.fixture
def get_fake_ir_engine():
    model_xml = 'model1.xml'
    model_bin = 'model1.bin'
    exec_net = None
    input_key = 'input'
    inputs = {input_key: []}
    outputs = ['output']
    engine = IrEngine(model_bin=model_bin, model_xml=model_xml,
                      exec_net=exec_net, inputs=inputs, outputs=outputs)

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
