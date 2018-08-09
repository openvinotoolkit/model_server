from concurrent import futures
import time
import grpc
import tensorflow.contrib.util as tf_contrib_util
from tensorflow.core.framework import types_pb2
import numpy as np
from ie_serving.tensorflow_serving_api import prediction_service_pb2
from ie_serving.server.service import PredictionServiceServicer

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
    print("server start")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
