# grpcclient namespace

## Functions:

`TensorProto` processing:
- [make_tensor_proto](make_tensor_proto.md)
- [make_ndarray](make_ndarray.md)

Creating server requests:
- [make_predict_request](make_grpc_predict_request.md) - alias for [make_grpc_predict_request](make_grpc_client.md)
- [make_metadata_request](make_grpc_metadata_request.md) - alias for [make_grpc_metadata_request](make_grpc_client.md)
- [make_status_request](make_grpc_status_request.md) - alias for [make_grpc_status_request](make_grpc_client.md)

Creating clients: 
- [make_client](make_grpc_client.md) - alias for [make_grpc_client](make_grpc_client.md)

## Usage:

```
from ovmsclient import grpcclient

client = grpcclient.make_client({"address": "localhost", "port": 9000})

proto = grpcclient.make_tensor_proto([1,2,3])
request = grpcclient.make_predict_request({"input": proto}, "model")

response = client.predict(request)
```
