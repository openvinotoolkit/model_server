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

import zmq
import os
import numpy as np

from ie_serving.config import GLOBAL_CONFIG

from multiprocessing import shared_memory
from ie_serving.messaging.endpoint_responses_pb2 import EndpointResponse, \
    PredictResponse
from ie_serving.messaging.data_attributes_pb2 import NumpyAttributes
from ie_serving.messaging.endpoint_requests_pb2 import EndpointRequest
import datetime

outputs = {"output": np.zeros((1, 1000), dtype=np.float32)}

def prepare_ipc_endpoint_response():
    ipc_endpoint_response = EndpointResponse()
    ipc_predict_response = PredictResponse()
    ipc_outputs = []

    for output_name in list(outputs.keys()):
        single_output = outputs[output_name]

        output_shm = shared_memory.SharedMemory(create=True,
                                                size=single_output.nbytes)
        shm_array = np.ndarray(single_output.shape, dtype=single_output.dtype,
                               buffer=output_shm.buf)
        shm_array[:] = single_output

        ipc_numpy_attributes = NumpyAttributes()
        ipc_numpy_attributes.shape.extend(list(shm_array.shape))
        ipc_numpy_attributes.data_type = shm_array.dtype.name

        ipc_output_data = PredictResponse.Data()
        ipc_output_data.numpy_attributes.CopyFrom(ipc_numpy_attributes)
        ipc_output_data.output_name = output_name
        ipc_output_data.shm_name = output_shm.name
        ipc_outputs.append(ipc_output_data)

    ipc_predict_response.outputs.extend(ipc_outputs)
    ipc_predict_response.responding_version = 1
    ipc_endpoint_response.predict_response.CopyFrom(ipc_predict_response)
    return ipc_endpoint_response


def free_inputs_shm(ipc_predict_request):
    for ipc_input in ipc_predict_request.inputs:
        shm = shared_memory.SharedMemory(name=ipc_input.shm_name)
        shm.close()
        shm.unlink()


def run_fake_engine():
    zmq_context = zmq.Context()
    engine_socket_name = os.path.join(GLOBAL_CONFIG['tmp_files_dir'],
                                      "{}-{}.sock".format("fake-model", 1))
    engine_socket = zmq_context.socket(zmq.REP)
    engine_socket.bind("ipc://{}".format(engine_socket_name))

    print("Starting listening for inference requests")
    while True:
        start_time = datetime.datetime.now()
        ipc_raw_request = engine_socket.recv()
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        #print("Awaiting request time - {} ms".format(duration))
        start_time = datetime.datetime.now()
        ipc_endpoint_response = prepare_ipc_endpoint_response()
        ipc_endpoint_request = EndpointRequest()
        ipc_endpoint_request.MergeFromString(ipc_raw_request)
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        #print("Msg preparation and deserialization - {} ms".format(duration))
        #print(ipc_endpoint_request)
        # TODO: validate request
        engine_socket.send(b'ACK')
        return_socket_name = ipc_endpoint_request.predict_request.\
            return_socket_name
        return_socket = zmq_context.socket(zmq.REQ)
        return_socket.connect("ipc://{}".format(return_socket_name))
        start_time = datetime.datetime.now()
        msg = ipc_endpoint_response.SerializeToString()
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        #print("Output serialization - {} ms".format(duration))
        return_socket.send(msg)
        start_time = datetime.datetime.now()
        return_socket.recv()
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        #print("Receiving confirmation - {} ms".format(duration))

        start_time = datetime.datetime.now()
        free_inputs_shm(ipc_endpoint_request.predict_request)
        duration = (datetime.datetime.now()
                    - start_time).total_seconds() * 1000
        #print("Memory cleanup - {} ms".format(duration))

# This script imitates inference engine with a name "fake-model", version 1.
# It listens for requests from the server process and sends back a valid
# message. To run it with docker container, mount this script and run it
# alongside model server.


if __name__ == "__main__":
    run_fake_engine()
