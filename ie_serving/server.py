from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from concurrent import futures
import time
import grpc
import tensorflow as tf
from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_shape
from inference_engine import IENetwork, IEPlugin
import datetime
import argparse
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

w = 224
h = 224


class PredictionServiceServicer(prediction_service_pb2.BetaPredictionServiceServicer):

    def Predict(self, request, context):
        """
        Predict -- provides access to loaded TensorFlow model.
        """
        start_time = datetime.datetime.now()
        input = tf.contrib.util.make_ndarray(request.inputs['in'])
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print("shape", input.shape)
        print("input reading time in ms:", int(duration.total_seconds() * 1000))
        start_time = datetime.datetime.now()
        test = self.Infer(input)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print("inference time in ms:", int(duration.total_seconds() * 1000))
        response = predict_pb2.PredictResponse()
        start_time = datetime.datetime.now()

        testowo = tensorflow_dot_core_dot_framework_dot_tensor__pb2.TensorProto(
            dtype=types_pb2.DT_FLOAT,
            tensor_shape=tensor_shape.as_shape(test['resnet_v1_50/predictions/Reshape_1'].shape).as_proto())
        for result in test['resnet_v1_50/predictions/Reshape_1']:
            testowo.float_val.extend(result)
        response.outputs['out'].CopyFrom(testowo)

        '''
        response.outputs['out'].CopyFrom(
        tf.contrib.util.make_tensor_proto(test['resnet_v1_50/predictions/Reshape_1'],
                                          shape=test['resnet_v1_50/predictions/Reshape_1'].shape,
                                          dtype=types_pb2.DT_FLOAT))

        '''
        response.model_spec.name = 'resnet'
        response.model_spec.version.value = 1
        response.model_spec.signature_name = "serving_default"
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print("preparing tensor_proto time in ms:", int(duration.total_seconds() * 1000))
        return response


    def Infer(self,image):
        image = image.transpose((0, 3, 1, 2))  # NHWC -> NCHW
        results = exec_net.infer(inputs={input_blob: image})
        return results


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_service_pb2.add_PredictionServiceServicer_to_server(PredictionServiceServicer(), server)
    server.add_insecure_port('[::]:50000')
    server.start()
    print("server start")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gRPC server for inference')
    parser.add_argument('--model', required=True, help='path to IR model without extension')
    args = vars(parser.parse_args())

    device = "CPU"
    plugin_dir = None
    model_xml = args['model'] + ".xml"
    model_bin = args['model'] + ".bin"
    cpu_extension = "/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so"

    plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
    if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)

    print("inference engine:", model_xml, model_bin, device)
    # Read IR
    print("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)


    print(net)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    test_list = np.zeros((1,1),dtype=float)
    # its only to initalize tensorflow
    tf.contrib.util.make_tensor_proto(test_list,
                                      shape=test_list.shape,
                                      dtype=types_pb2.DT_FLOAT)
    print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=1)
    print("input blobs", net.inputs[input_blob])
    serve()
