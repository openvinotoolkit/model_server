from openvino.inference_engine import IECore
import numpy as np
import datetime
import time
import statistics

xml_path = "/home/mzeglars/models/ssd_mobilenet_v2_coco/1/ssd_mobilenet_v2_coco.xml"
bin_path = "/home/mzeglars/models/ssd_mobilenet_v2_coco/1/ssd_mobilenet_v2_coco.bin"

xml_path = "/home/mzeglars/models/resnet-50-tf/1/resnet-50-tf.xml"
bin_path = "/home/mzeglars/models/resnet-50-tf/1/resnet-50-tf.bin"

core = IECore()
network = core.read_network(xml_path, bin_path)
exec_network = core.load_network(network, "CPU")

input = { "image_tensor": np.random.rand(1,3,300,300).astype(np.float32) }
input = { "map/TensorArrayStack/TensorArrayGatherV3": np.random.rand(1,3,224,224).astype(np.float32) }

iterations = 1000
execution_times = []
for i in range(iterations):
    time.sleep(0.1)
    start_time = datetime.datetime.now()
    exec_network.requests[0].async_infer(input)
    exec_network.requests[0].wait()
    end_time = datetime.datetime.now()
    execution_times.append((end_time - start_time).total_seconds() * 1000.)


print("Average latency: {} ms".format(statistics.mean(execution_times)))
