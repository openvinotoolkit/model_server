from ie_serving.tensorflow_serving_api import prediction_service_pb2
from concurrent import futures
import time
import grpc
from ie_serving.server.predict import PredictionServiceServicer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def serve(models, max_workers: int=10, port: int=9001):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    prediction_service_pb2.add_PredictionServiceServicer_to_server(PredictionServiceServicer(models=models), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    print("server start")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
