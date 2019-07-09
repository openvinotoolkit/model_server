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

from concurrent import futures
import time
import grpc
import sys
import tensorflow.contrib.util as tf_contrib_util
from tensorflow.core.framework import types_pb2
from cheroot.wsgi import Server as WSGIServer, PathInfoDispatcher
import numpy as np

from ie_serving.models.models_utils import statuses_as_dicts
from ie_serving.tensorflow_serving_api import prediction_service_pb2
from ie_serving.server.service import PredictionServiceServicer
from ie_serving.logger import get_logger
from ie_serving.config import FILE_SYSTEM_POLL_WAIT_SECONDS
from ie_serving.server.rest_service import create_rest_api

logger = get_logger(__name__)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
GIGABYTE = 1024 ** 3


def initialize_tf():
    initialization_list = np.zeros((1, 1), dtype=float)
    tf_contrib_util.make_ndarray(initialization_list,
                                 shape=initialization_list.shape,
                                 dtype=types_pb2.DT_FLOAT)
    pass


def serve(models, max_workers: int=1, port: int=9000):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                         options=[('grpc.max_send_message_length', GIGABYTE),
                                  ('grpc.max_receive_message_length', GIGABYTE)
                                  ])
    prediction_service_pb2.add_PredictionServiceServicer_to_server(
        PredictionServiceServicer(models=models), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    logger.info("Server listens on port {port} and will be "
                "serving models: {models}".format(port=port,
                                                  models=list(models.keys())))
    try:
        while True:
            time.sleep(FILE_SYSTEM_POLL_WAIT_SECONDS)
            for model in models:
                models[model].update()
                logger.info(statuses_as_dicts(
                    models[model].versions_statuses.values()))
    except KeyboardInterrupt:
        server.stop(0)
        sys.exit(0)


def start_web_rest_server(models, rest_port):
    d = PathInfoDispatcher({'/': create_rest_api(models)})
    server = WSGIServer(('0.0.0.0', rest_port), d,
                        numthreads=1, request_queue_size=50)

    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
