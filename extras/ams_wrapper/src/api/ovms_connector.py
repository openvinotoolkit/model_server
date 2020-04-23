import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class OvmsConnector():
    def __init__(self, ovms_port, ovms_model_info):
        self.ovms_port = ovms_port
        self.model_name = ovms_model_info['model_name']
        self.model_version = ovms_model_info['model_version']
        self.input_name = ovms_model_info['input_name']
        self.input_shape = ovms_model_info['input_shape']

        channel = grpc.insecure_channel("{}:{}".format("127.0.0.1", self.ovms_port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    def send(self, inference_input):
        # TODO: prepare request and handle response
        request = predict_pb2.PredictRequest()
        result = self.stub.Predict(request, 10.0)
        return 