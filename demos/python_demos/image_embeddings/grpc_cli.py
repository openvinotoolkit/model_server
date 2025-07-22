import sys
sys.path.append("../../common/python")
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
from tritonclient.utils import serialize_byte_tensor

import argparse
import datetime
import numpy as np
from client_utils import print_statistics
from urllib.request import urlretrieve
from pathlib import Path
import os
import grpc
import time

parser = argparse.ArgumentParser(description='GRPC client for clip example')

parser.add_argument('--timeout', required=False, default='15',
                    help='Specify timeout to wait for models readiness on the server in seconds. default 15 seconds.')
parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')

parser.add_argument('--image_url', required=False, default='https://images.unsplash.com/photo-1716467891152-1b43a96de578?q=80&w=1481&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
                    help='Specify image_url to send to the CLIP model. default:https://images.unsplash.com/photo-1716467891152-1b43a96de578?q=80&w=1481&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')

args = vars(parser.parse_args())

iterations = args.get('iterations')
iteration = 0

timeout = int(args.get('timeout'))
# Check models ready
client = grpcclient.InferenceServerClient(args['url'])
channel = grpc.insecure_channel(args['url'])
grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

while(timeout):
    request = service_pb2.ServerReadyRequest()
    response = grpc_stub.ServerReady(request)
    print("Server Ready: {}".format(response.ready))
    if response.ready:
        break
    time.sleep(1)
    timeout-=1

if not response.ready:
    print("Models are not ready. Increase timeout or check server setup and errors.")
    exit(-1)

image_url = args['image_url']
print(f"Using image_url:\n{image_url}\n")

input_name = image_url.split("/")[-1]
sample_path = Path(os.path.join("data", input_name))
if not os.path.exists(sample_path):
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        image_url,
        sample_path,
    )

with open(sample_path, "rb") as f:
    image_data = f.read()

image_np = np.array([image_data], dtype=np.object_)

# Set up inference input correctly
image_input = grpcclient.InferInput("image", [1], "BYTES")
image_input.set_data_from_numpy(image_np)

processing_times = []
start_time = datetime.datetime.now()
results = client.infer("python_model", [image_input])
end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds() * 1000
processing_times.append(int(duration))
print(f"Detection:\n{results.as_numpy('embedding')}\n")

print_statistics(np.array(processing_times,int), batch_size = 1)

