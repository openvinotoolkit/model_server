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

import zmq
import os
import numpy as np


from ie_serving.config import GLOBAL_CONFIG
from ie_serving.logger import get_logger


from tensorflow import __version__ as tf_version
if tf_version.split(".")[0] == "2":
    from tensorflow import make_ndarray, make_tensor_proto
else:  # TF version 1.x
    from tensorflow.contrib.util import make_ndarray, make_tensor_proto

from multiprocessing import shared_memory
from ie_serving.messaging.endpoint_responses_pb2 import EndpointResponse, PredictResponse
from ie_serving.messaging.data_attributes_pb2 import NumpyAttributes
from ie_serving.messaging.endpoint_requests_pb2 import EndpointRequest, PredictRequest

def prepare_ipc_predict_response():
    outputs = {"output": np.zeros((1,100))}
    ipc_endpoint_response = EndpointResponse()
    ipc_predict_response = PredictResponse()
    allocated_shm_names = []
    ipc_outputs = []

    for output_name in list(outputs.keys()):
        single_output = outputs[output_name]

        output_shm = shared_memory.SharedMemory(create=True,
                                                size=single_output.nbytes)
        shm_array = np.ndarray(single_output.shape, dtype=single_output.dtype,
                               buffer=output_shm.buf)
        shm_array[:] = single_output[:]

        ipc_numpy_attributes = NumpyAttributes()
        ipc_numpy_attributes.shape.extend(list(shm_array.shape))
        ipc_numpy_attributes.data_type = shm_array.dtype.name

        ipc_output_data = PredictResponse.Data()
        ipc_output_data.numpy_attributes.CopyFrom(ipc_numpy_attributes)
        ipc_output_data.output_name = output_name
        ipc_output_data.shm_name = output_shm.name
        ipc_outputs.append(ipc_output_data)

        allocated_shm_names.append(output_shm.name)

    ipc_predict_response.outputs.extend(ipc_outputs)
    ipc_predict_response.responding_version = 1
    ipc_endpoint_response.predict_response.CopyFrom(ipc_predict_response)
    return ipc_endpoint_response


def run_fake_engine():
    prepare_ipc_predict_response()
    zmq_context = zmq.Context()
    engine_socket_name = os.path.join(GLOBAL_CONFIG['tmp_files_dir'],
                                      "{}-{}.sock".format("fake-model", 1))
    engine_socket = zmq_context.socket(zmq.REP)
    engine_socket.bind("ipc://{}".format(engine_socket_name))

    print("Starting listening for inference requests")
    while True:
        print("Awaiting request...")
        ipc_predict_response = prepare_ipc_predict_response()
        ipc_endpoint_request = EndpointRequest()
        ipc_endpoint_request.MergeFromString(engine_socket.recv())
        print(ipc_endpoint_request)
        # TODO: validate request
        engine_socket.send(b'ACK')
        return_socket_name = ipc_endpoint_request.predict_request.\
            return_socket_name
        return_socket = zmq_context.socket(zmq.REQ)
        return_socket.connect("ipc://{}".format(return_socket_name))
        return_socket.send(ipc_predict_response.SerializeToString())
        return_socket.recv()

# This script imitates inference engine with a name "fake-model", version 1.
# It listens for requests from the server process and sends back a valid message.
# To run it with docker container, mount this script and run it alongside model server.
#
if __name__ == "__main__":
    run_fake_engine()