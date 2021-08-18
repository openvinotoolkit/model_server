# OVMS Python Client Library API

## Functions:

`TensorProto` processing:
- [make_tensor_proto](make_tensor_proto.md)
- [make_ndarray](make_ndarray.md)

Creating server requests:
- [make_grpc_predict_request](make_grpc_predict_request.md)
- [make_grpc_metadata_request](make_grpc_metadata_request.md)
- [make_grpc_status_request](make_grpc_status_request.md)

Creating clients: 
- [make_grpc_client](make_grpc_client.md)


*Note*: Above functions are also aliased in the following namespaces:
 - [grpcclient](grpcclient.md)


---

## Classes:

Client classes:
- [GrpcClient](grpc_client.md)

Server response classes:
- [GrpcPredictResponse](grpc_predict_response.md)
- [GrpcModelMetadataResponse](grpc_model_metadata_response.md)
- [GrpcModelStatusResponse](grpc_model_status_response.md)
