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

import datetime
import numpy as np

from multiprocessing import shared_memory
from ie_serving.messaging.endpoint_requests_pb2 import EndpointRequest, \
    PredictRequest
from ie_serving.messaging.data_attributes_pb2 import NumpyAttributes

from tensorflow import __version__ as tf_version
if tf_version.split(".")[0] == "2":
    from tensorflow import make_ndarray, make_tensor_proto
else:  # TF version 1.x
    from tensorflow.contrib.util import make_ndarray, make_tensor_proto

from ie_serving.logger import get_logger
logger = get_logger(__name__)

def extract_inference_output(ipc_predict_response):
    inference_output = {}
    for output in ipc_predict_response.outputs:
        output_shm = shared_memory.SharedMemory(name=output.shm_name)
        output_results = np.ndarray(
            shape=tuple(output.numpy_attributes.shape),
            dtype=np.dtype(output.numpy_attributes.data_type),
            buffer=output_shm.buf)
        inference_output[output.output_name] = output_results
    return inference_output

def prepare_ipc_predict_request(data_type, data, return_socket_name):
    # TODO: handling various data types
    ipc_endpoint_request = EndpointRequest()
    ipc_predict_request = PredictRequest()
    ipc_inputs = []

    inputs = dict(data)
    for input_name in list(inputs.keys()):
        start_time = datetime.datetime.now()
        single_input = make_ndarray(inputs[input_name])
        duration = (datetime.datetime.now() -start_time).total_seconds() * 1000
        logger.debug("Numpy deserialization: - {} ms".format(duration))

        start_time = datetime.datetime.now()
        input_shm = shared_memory.SharedMemory(create=True,
                                               size=single_input.nbytes)
        shm_array = np.ndarray(single_input.shape, dtype=single_input.dtype,
                               buffer=input_shm.buf)
        duration = (datetime.datetime.now() -start_time).total_seconds() * 1000
        logger.debug("Shared memory allocation - {} ms".format(duration))

        start_time = datetime.datetime.now()
        shm_array[:] = single_input[:]
        duration = (datetime.datetime.now() -start_time).total_seconds() * 1000
        logger.debug("Input data copying to shared memory - {} ms".format(
            duration))

        start_time = datetime.datetime.now()
        ipc_numpy_attributes = NumpyAttributes()
        ipc_numpy_attributes.shape.extend(list(shm_array.shape))
        ipc_numpy_attributes.data_type = shm_array.dtype.name

        ipc_input_data = PredictRequest.Data()
        ipc_input_data.numpy_attributes.CopyFrom(ipc_numpy_attributes)
        ipc_input_data.input_name = input_name
        ipc_input_data.shm_name = input_shm.name
        ipc_inputs.append(ipc_input_data)
        duration = (datetime.datetime.now() -start_time).total_seconds() * 1000
        logger.debug("Single input IPC message preparation - {} ms".format(
            duration))

    start_time = datetime.datetime.now()
    ipc_predict_request.return_socket_name = return_socket_name
    ipc_predict_request.inputs.extend(ipc_inputs)
    ipc_endpoint_request.predict_request.CopyFrom(ipc_predict_request)
    duration = (datetime.datetime.now() -start_time).total_seconds() * 1000
    logger.debug("Final request IPC message preparation - {} ms".format(
        duration))
    return ipc_endpoint_request