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

from concurrent import futures
from cheroot.wsgi import Server as WSGIServer, PathInfoDispatcher
import grpc
import numpy as np
import sys
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time

from ie_serving.config import GLOBAL_CONFIG
from ie_serving.logger import get_logger
from ie_serving.server.rest_service import create_rest_api
from ie_serving.server.service import PredictionServiceServicer, \
    ModelServiceServicer

from tensorflow import __version__ as tf_version
if tf_version.split(".")[0] == "2":
    from tensorflow import make_ndarray
else:  # TF version 1.x
    from tensorflow.contrib.util import make_ndarray


logger = get_logger(__name__)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
GIGABYTE = 1024 ** 3


def initialize_tf():
    initialization_list = np.zeros((1, 1), dtype=float)
    make_ndarray(initialization_list, shape=initialization_list.shape,
                 dtype=types_pb2.DT_FLOAT)
    pass


def serve(models, max_workers: int=1, port: int=9000):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                         options=[('grpc.max_send_message_length', GIGABYTE),
                                  ('grpc.max_receive_message_length', GIGABYTE)
                                  ])
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServiceServicer(models=models), server)
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceServicer(models=models), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    logger.info("gRPC server listens on port {port} and will be "
                "serving models: {models}".format(port=port,
                                                  models=list(models.keys())))
    try:
        while True:
            if GLOBAL_CONFIG['file_system_poll_wait_seconds'] > 0:
                time.sleep(GLOBAL_CONFIG['file_system_poll_wait_seconds'])
                for model in models:
                    models[model].update()
            else:
                time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        sys.exit(0)


def start_web_rest_server(models, rest_port, num_threads):
    d = PathInfoDispatcher({'/': create_rest_api(models)})
    server = WSGIServer(('0.0.0.0', rest_port), d,
                        numthreads=num_threads,
                        request_queue_size=GLOBAL_CONFIG[
                            'rest_requests_queue_size'])
    logger.info("REST server listens on port {port} and will be "
                "serving models: {models}".format(port=rest_port,
                                                  models=list(models.keys())))
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
