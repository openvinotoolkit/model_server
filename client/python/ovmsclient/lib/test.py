from ovmsclient import make_grpc_client
from ovmsclient.tfs_compat.http.requests import make_predict_request
import numpy as np
#with open("../samples/images/zebra.jpeg", "rb") as f:
#    img = f.read()
#inputs = {"map/TensorArrayStack/TensorArrayGatherV3": img}

#inputs = {"map/TensorArrayStack/TensorArrayGatherV3": bytes([1,2,3])}
client = make_grpc_client("localhost:9000")

#request = make_predict_request(inputs, "resnet")
#print(request.parsed_inputs)

inputs = {"input": np.zeros((1,300,300,300), dtype=np.float32)}

client.predict(inputs, "resnet")

