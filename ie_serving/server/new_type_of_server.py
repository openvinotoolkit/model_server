from ie_serving.tensorflow_serving_api import prediction_service_pb2, predict_pb2
from concurrent import futures
import time
import grpc
from inference_engine import IENetwork, IEPlugin
from ie_serving.server.predict import PredictionServiceServicer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def serve(max_workers: int=10, port: int=9000):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    prediction_service_pb2.add_PredictionServiceServicer_to_server(PredictionServiceServicer(), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    device = "CPU"
    plugin_dir = None
    model_xml = "/home/marek/models/frozen_resnet_v1_50" + ".xml"
    model_bin = "/home/marek/models/frozen_resnet_v1_50" + ".bin"
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

    print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=1)
    print("input blobs", net.inputs[input_blob])
    serve(max_workers=10, port=50000)
