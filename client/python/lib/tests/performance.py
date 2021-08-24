import ovmsclient
import tensorflow
import datetime
import numpy as np
from ovmsclient.tfs_compat.grpc.requests import make_predict_request
from ovmsclient.tfs_compat.grpc.tensors import make_tensor_proto
from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client
from tensorflow.core.framework.types_pb2 import DataType
from tensorflow_serving.apis import predict_pb2
import pandas as pd

np.random.seed(0)

# parameters
# iterations determine how many times tested method should be called
iterations = 1_000_000
# lower_iterations determine how many times tested method should be called for
# time-consuming tests
lower_iterations = 10_000
# for make_* functions from ovmsclient performance testing enable desired test
# for TF impementations testing import tensorflow instead of ovmsclient and change
# function calls from ovmsclient.* to tensorflow.*
make_tensor_proto_testing = False
make_ndarray_testing = False
# for predict_request creation testing there are 2 different cases
# for make_predict_request testing import ovmsclient and do not
# import tensorflow and enable the test below
make_predict_request_testing = False
# for predict request creation testing via TF import tensorflow instead of ovmsclient
predict_request_TF_testing = False
# for predict testing on ovmsclient import ovmsclient and do not
# import tensorflow and enable the test below
predict_ovmsclient = False
# for predict testing via TF import tensorflow instead of ovmsclient
predict_TF = False
# for ovmsclient flow testing on ovmsclient import ovmsclient and do not
# import tensorflow and enable the test below
ovmsclient_testing = False
# for tf flow testing import tensorflow instead of ovmsclient
tf_testing = True
# testing protos apply to make_tensor_proto, make_ndarray and make_predict_request tests
proto_inputs_dict = {
    "float_shape_dtype_no_reshape": {
        "values": 5.0,
        "shape": [1],
        "dtype": DataType.DT_FLOAT
    },
    "float_no_shape_dtype_no_reshape": {
        "values": 5.0,
        "dtype": DataType.DT_FLOAT
    },
    "bytes_shape_dtype_no_reshape": {
        "values": bytes([1, 2, 3]),
        "shape": [1],
        "dtype": DataType.DT_STRING
    },
    "bytes_no_shape_dtype_no_reshape": {
        "values": bytes([1, 2, 3]),
        "dtype": DataType.DT_STRING
    },
    "list_shape_dtype_no_reshape": {
        "values": [1, 2, 3],
        "shape": [3],
        "dtype": DataType.DT_INT64
    },
    "list_no_shape_dtype_no_reshape": {
        "values": [1, 2, 3],
        "dtype": DataType.DT_INT64
    },
    "2Dlist_shape_dtype_no_reshape": {
        "values": [[1, 2, 3], [1, 2, 3]],
        "shape": [2, 3],
        "dtype": DataType.DT_INT64
    },
    "2Dlist_no_shape_dtype_no_reshape": {
        "values": [[1, 2, 3], [1, 2, 3]],
        "dtype": DataType.DT_INT64
    },
    "binary_shape_dtype_no_reshape": {
        "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
        "shape": [2],
        "dtype": DataType.DT_STRING
    },
    "binary_no_shape_dtype_no_reshape": {
        "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
        "dtype": DataType.DT_STRING
    },
    # "binary_to_dtype_shape_dtype_no_reshape": {   # need to be comment for tensorflow
    #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    #     "shape": [6],
    #     "dtype": DataType.DT_INT8
    # },
    # "binary_to_dtype_no_shape_dtype_no_reshape": {    # need to be comment for tensorflow
    #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    #     "dtype": DataType.DT_INT8
    # },
    "ndarray_shape_dtype_no_reshape": {
        "values": np.array([1, 2, 3]),
        "shape": [3],
        "dtype": DataType.DT_FLOAT
    },
    "ndarray_no_shape_dtype_no_reshape": {
        "values": np.array([1, 2, 3]),
        "dtype": DataType.DT_FLOAT
    },
    "2Dndarray_shape_dtype_no_reshape": {
        "values": np.array([[1, 2, 3], [1, 2, 3]]),
        "shape": [2, 3],
        "dtype": DataType.DT_FLOAT
    },
    "2Dndarray_no_shape_dtype_no_reshape": {
        "values": np.array([[1, 2, 3], [1, 2, 3]]),
        "dtype": DataType.DT_FLOAT
    },
    "float_shape_no_dtype_no_reshape": {
        "values": 5.0,
        "shape": [1],
    },
    "float_no_shape_no_dtype_no_reshape": {
        "values": 5.0,
    },
    "bytes_shape_no_dtype_no_reshape": {
        "values": bytes([1, 2, 3]),
        "shape": [1],
    },
    "bytes_no_shape_no_dtype_no_reshape": {
        "values": bytes([1, 2, 3]),
    },
    "list_shape_no_dtype_no_reshape": {
        "values": [1, 2, 3],
        "shape": [3]
    },
    "list_no_shape_no_dtype_no_reshape": {
        "values": [1, 2, 3],
    },
    "2Dlist_shape_no_dtype_no_reshape": {
        "values": [[1, 2, 3], [1, 2, 3]],
        "shape": [2, 3],
    },
    "2Dlist_no_shape_no_dtype_no_reshape": {
        "values": [[1, 2, 3], [1, 2, 3]],
    },
    "binary_shape_no_dtype_no_reshape": {
        "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
        "shape": [2],
    },
    "binary_no_shape_no_dtype_no_reshape": {
        "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    },
    # "binary_to_dtype_shape_no_dtype_no_reshape": {    #impossible
    # },
    # "binary_to_dtype_no_shape_no_dtype_no_reshape": { #impossible
    # },
    "ndarray_shape_no_dtype_no_reshape": {
        "values": np.array([1.0, 2.0, 3.0]),
        "shape": [3],
    },
    "ndarray_no_shape_no_dtype_no_reshape": {
        "values": np.array([1.0, 2.0, 3.0]),
    },
    "2Dndarray_shape_no_dtype_no_reshape": {
        "values": np.array([[1, 2, 3], [1, 2, 3]]),
        "shape": [2, 3],
    },
    "2Dndarray_no_shape_no_dtype_no_reshape": {
        "values": np.array([[1, 2, 3], [1, 2, 3]]),
    },
    # "float_shape_dtype_reshape": {   #impossible
    # },
    # "float_no_shape_dtype_reshape": {    #impossible
    # },
    # "list_shape_dtype_reshape": { #impossible
    # },
    # "list_no_shape_dtype_reshape": {  #impossible
    # },
    "2Dlist_shape_dtype_reshape": {
        "values": [[1, 2, 3], [1, 2, 3]],
        "shape": [6],
        "dtype": DataType.DT_INT64
    },
    # "2Dlist_no_shape_dtype_reshape": {    #impossible
    # },
    # "binary_shape_dtype_reshape": {   #impossible
    # },
    # "binary_no_shape_dtype_reshape": {    #impossible
    # },
    # "ndarray_shape_dtype_reshape": {  #impossible
    # },
    # "ndarray_no_shape_dtype_reshape": {   #impossible
    # },
    "2Dndarray_shape_dtype_reshape": {
        "values": np.array([[1, 2, 3], [1, 2, 3]]),
        "shape": [6],
        "dtype": DataType.DT_FLOAT
    },
    # "2Dndarray_no_shape_no_dtype_reshape": {   #impossible
    # },
    # "scalar_shape_no_dtype_reshape": {    #impossible
    # },
    # "scalar_no_shape_no_dtype_reshape": { #impossible
    # },
    # "bytes_shape_no_dtype_reshape": { #impossible
    # },
    # "bytes_no_shape_no_dtype_reshape": {  #impossible
    # },
    # "list_shape_no_dtype_reshape": {  #impossible
    # },
    # "list_no_shape_no_dtype_reshape": {   #impossible
    # },
    "2Dlist_shape_no_dtype_reshape": {
        "values": [[1, 2, 3], [1, 2, 3]],
        "shape": [6],
    },
    # "2Dlist_no_shape_no_dtype_reshape": {   #impossible
    # },
    # "binary_shape_no_dtype_reshape": {    #impossible
    # },
    # "binary_no_shape_no_dtype_reshape": { #impossible
    # },
    # "binary_to_dtype_shape_no_dtype_reshape": {   #impossible
    # },
    # "binary_to_dtype_no_shape_no_dtype_reshape": {    #impossible
    # },
    # "ndarray_shape_no_dtype_reshape": {   #impossible
    # },
    # "ndarray_no_shape_no_dtype_reshape": {    #impossible
    # },
    "2Dndarray_shape_no_dtype_reshape": {
        "values": np.array([[1, 2, 3], [1, 2, 3]]),
        "shape": [6],
    },
    # "2Dndarray_no_shape_no_dtype_reshape": {    #impossible
    # },
    "ndarray_zeros_resnet_npdtype_dtype_no_reshape": {
        "values": np.zeros((1, 3, 244, 244), dtype=np.float32),
        "dtype": DataType.DT_FLOAT
    },
    "ndarray_zeros_resnet_no_npdtype_dtype_no_reshape": {
        "values": np.zeros((1, 3, 244, 244)),
        "dtype": DataType.DT_INT16
    },
    "ndarray_zeros_resnet_npdtype_no_dtype_no_reshape": {
        "values": np.zeros((1, 3, 244, 244), dtype=np.float32),
    },
    "ndarray_zeros_resnet_no_npdtype_no_dtype_no_reshape": {
        "values": np.zeros((1, 3, 244, 244)),
    },
    "ndarray_zeros_resnet_npdtype_dtype_reshape": {
        "values": np.zeros((1, 3, 244, 244), dtype=np.float32),
        "dtype": DataType.DT_FLOAT,
        "shape": [1, 2, 24, 3721],
    },
    "ndarray_zeros_resnet_no_npdtype_dtype_reshape": {
        "values": np.zeros((1, 3, 244, 244)),
        "dtype": DataType.DT_INT16,
        "shape": [1, 2, 24, 3721],
    },
    "ndarray_zeros_resnet_npdtype_no_dtype_reshape": {
        "values": np.zeros((1, 3, 244, 244), dtype=np.float32),
        "shape": [1, 2, 24, 3721],
    },
    "ndarray_zeros_resnet_no_npdtype_no_dtype_reshape": {
        "values": np.zeros((1, 3, 244, 244)),
        "shape": [1, 2, 24, 3721],
    },
    "ndarray_rand_resnet_npdtype_dtype": {
        "values": np.random.rand(1, 3, 244, 244).astype(np.float32),
        "dtype": DataType.DT_FLOAT,
    },
    "ndarray_rand_resnet_no_npdtype_dtype": {
        "values": np.random.rand(1, 3, 244, 244),
        "dtype": DataType.DT_FLOAT,
    },
    "ndarray_rand_resnet_npdtype_no_dtype": {
        "values": np.random.rand(1, 3, 244, 244).astype(np.float32),
    },
    "ndarray_rand_resnet_no_npdtype_no_dtype": {
        "values": np.random.rand(1, 3, 244, 244),
    },
    "Big_2Dlist_shape_dtype_no_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
        "shape": [3, 2, 5000],
        "dtype": DataType.DT_INT64
    },
    "Big_2Dlist_no_shape_dtype_no_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
        "dtype": DataType.DT_INT64
    },
    "Big_2Dlist_shape_no_dtype_no_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
        "shape": [3, 2, 5000],
    },
    "Big_2Dlist_no_shape_no_dtype_no_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
    },
    "Big_2Dlist_shape_dtype_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
        "shape": [2, 6, 2500],
        "dtype": DataType.DT_INT64
    },
    # "Big_2Dlist_no_shape_dtype_reshape": {    #impossible
    # },
    "Big_2Dlist_shape_no_dtype_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
        "shape": [2, 6, 2500],
    },
    # "Big_2Dlist_no_shape_no_dtype_reshape": {   #impossible
    # },
    "Big_binary_shape_dtype_no_reshape": {
        "values": [bytes([i % 256 for i in range(10000)]),
                   bytes([(3*i) % 256 for i in range(10000)]),
                   bytes([(7*i) % 256 for i in range(10000)])],
        "shape": [3],
        "dtype": DataType.DT_STRING
    },
    "Big_binary_no_shape_dtype_no_reshape": {
        "values": [bytes([i % 256 for i in range(10000)]),
                   bytes([(3*i) % 256 for i in range(10000)]),
                   bytes([(7*i) % 256 for i in range(10000)])],
        "dtype": DataType.DT_STRING
    },
    "Big_binary_shape_no_dtype_no_reshape": {
        "values": [bytes([i % 256 for i in range(10000)]),
                   bytes([(3*i) % 256 for i in range(10000)]),
                   bytes([(7*i) % 256 for i in range(10000)])],
        "shape": [3],
    },
    "Big_binary_no_shape_no_dtype_no_reshape": {
        "values": [bytes([i % 256 for i in range(10000)]),
                   bytes([(3*i) % 256 for i in range(10000)]),
                   bytes([(7*i) % 256 for i in range(10000)])],
    },
}

lower_iteration_cases = [
    "Big_2Dlist_shape_dtype_no_reshape", "Big_2Dlist_no_shape_dtype_no_reshape",
    "Big_2Dlist_shape_no_dtype_no_reshape", "Big_2Dlist_no_shape_no_dtype_no_reshape",
    "Big_2Dlist_shape_dtype_reshape", "Big_2Dlist_shape_no_dtype_reshape"
]

# make_tensor_proto performance testing
if make_tensor_proto_testing:
    make_tensor_proto_performance = {}

    for key, value in proto_inputs_dict.items():
        final_time = 0
        iteration = iterations if key not in lower_iteration_cases else lower_iterations
        for i in range(iteration):
            start_time = datetime.datetime.now()
            proto = ovmsclient.make_tensor_proto(**value)
            end_time = datetime.datetime.now()
            final_time += (end_time - start_time).total_seconds()
        time_per_proto = final_time/iteration * 1000
        make_tensor_proto_performance[key] = time_per_proto
        print(f'{key} finished with time {final_time}s')
        print(f'Time spend per proto: {time_per_proto}ms')

    print(make_tensor_proto_performance)
    proto_df = pd.DataFrame(data=make_tensor_proto_performance, index=[0])
    proto_df.astype('float32')
    proto_df = proto_df.T
    proto_df.columns = ['time_per_proto[ms]']
    proto_df.to_excel('performance.xlsx', sheet_name='make_tensor_proto_perf')

# make_ndarray performance testing
if make_ndarray_testing:
    make_ndarray_performance = {}

    for key, value in proto_inputs_dict.items():
        final_time = 0
        iteration = iterations if key not in lower_iteration_cases else lower_iterations
        proto = ovmsclient.make_tensor_proto(**value)
        for i in range(iteration):
            start_time = datetime.datetime.now()
            array = ovmsclient.make_ndarray(proto)
            end_time = datetime.datetime.now()
            final_time += (end_time - start_time).total_seconds()
        time_per_ndarray = final_time/iteration * 1000
        make_ndarray_performance[key] = time_per_ndarray
        print(f'{key} finished with time {final_time}s')
        print(f'Time spend per ndarray: {time_per_ndarray}ms')

    print(make_ndarray_performance)
    ndarray_df = pd.DataFrame(data=make_ndarray_performance, index=[0])
    ndarray_df.astype('float32')
    ndarray_df = ndarray_df.T
    ndarray_df.columns = ['time_per_ndarray[ms]']
    ndarray_df.to_excel('performance.xlsx', sheet_name='make_ndarray_perf')

# make_predict_request performance testing
if make_predict_request_testing:
    make_predict_request_performance = {}

    for key, value in proto_inputs_dict.items():
        final_time = 0
        iteration = iterations
        inputs = {
            "proto": ovmsclient.make_tensor_proto(**value)
        }
        for i in range(iteration):
            start_time = datetime.datetime.now()
            request = ovmsclient.make_grpc_predict_request(inputs, 'name', 0)
            end_time = datetime.datetime.now()
            final_time += (end_time - start_time).total_seconds()
        time_per_request = final_time/iteration * 1000
        make_predict_request_performance[key] = time_per_request
        print(f'{key} finished with time {final_time}s')
        print(f'Time spend per request: {time_per_request}ms')

    print(make_predict_request_performance)
    proto_df = pd.DataFrame(data=make_predict_request_performance, index=[0])
    proto_df.astype('float32')
    proto_df = proto_df.T
    proto_df.columns = ['time_per_request[ms]']
    proto_df.to_excel('performance.xlsx', sheet_name='make_predict_request_perf')

if predict_request_TF_testing:
    predict_request_TF_performance = {}

    for key, value in proto_inputs_dict.items():
        final_time = 0
        iteration = iterations
        proto = ovmsclient.make_tensor_proto(**value)
        for i in range(iteration):
            start_time = datetime.datetime.now()
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'name'
            request.model_spec.version.value = 0
            request.inputs['proto'].CopyFrom(proto)
            end_time = datetime.datetime.now()
            final_time += (end_time - start_time).total_seconds()
        time_per_request = final_time/iteration * 1000
        predict_request_TF_performance[key] = time_per_request
        print(f'{key} finished with time {final_time}s')
        print(f'Time spend per request: {time_per_request}ms')

    print(predict_request_TF_performance)
    proto_df = pd.DataFrame(data=predict_request_TF_performance, index=[0])
    proto_df.astype('float32')
    proto_df = proto_df.T
    proto_df.columns = ['time_per_request[ms]']
    proto_df.to_excel('performance.xlsx', sheet_name='make_predict_request_TF_perf')


# predict testing configuration
predict_proto = make_tensor_proto(np.zeros((1, 3, 224, 224), dtype=np.float32))
predict_iterations = 10_000
key = 'resnet'
inputs = {
        "0": predict_proto
}
request = make_predict_request(inputs, key)

config = {
    "address": "localhost",
    "port": 9000
}
client = make_grpc_client(config)

if predict_ovmsclient:
    predict_ovmsclient_performance = {}

    final_time = 0
    for i in range(predict_iterations):
        start_time = datetime.datetime.now()
        client.predict(request)
        end_time = datetime.datetime.now()
        final_time += (end_time - start_time).total_seconds()
    time_per_predict = final_time/predict_iterations * 1000
    predict_ovmsclient_performance[key] = time_per_predict
    print(f'{key} finished with time {final_time}s')
    print(f'Time spend per predict: {time_per_predict}ms')

    print(predict_ovmsclient_performance)
    proto_df = pd.DataFrame(data=predict_ovmsclient_performance, index=[0])
    proto_df.astype('float32')
    proto_df = proto_df.T
    proto_df.columns = ['time_per_predict[ms]']
    proto_df.to_excel('performance.xlsx', sheet_name='predict_ovmsclient_perf')

if predict_TF:
    predict_TF_performance = {}

    final_time = 0
    stub = client.prediction_service_stub
    raw_request = request.raw_request
    for i in range(predict_iterations):
        start_time = datetime.datetime.now()
        stub.Predict(raw_request, 10.0)
        end_time = datetime.datetime.now()
        final_time += (end_time - start_time).total_seconds()
    time_per_predict = final_time/predict_iterations * 1000
    predict_TF_performance[key] = time_per_predict
    print(f'{key} finished with time {final_time}s')
    print(f'Time spend per iteration: {time_per_predict}ms')

    print(predict_TF_performance)
    proto_df = pd.DataFrame(data=predict_TF_performance, index=[0])
    proto_df.astype('float32')
    proto_df = proto_df.T
    proto_df.columns = ['time_per_predict[ms]']
    proto_df.to_excel('performance.xlsx', sheet_name='predict_TF_perf')

# whole flow testing
flow_iterations = 10_000

if ovmsclient_testing:
    ovmsclient_performance = {}

    final_time = 0
    for i in range(flow_iterations):
        start_time = datetime.datetime.now()
        proto = ovmsclient.make_tensor_proto(np.zeros((1, 3, 224, 224), dtype=np.float32))
        inputs = {
            "0": proto
        }
        request = ovmsclient.make_grpc_predict_request(inputs, 'resnet')
        response = client.predict(request)
        response_dict = response.to_dict()
        end_time = datetime.datetime.now()
        final_time += (end_time - start_time).total_seconds()
    time_per_iteration = final_time/flow_iterations * 1000
    ovmsclient_performance[key] = time_per_iteration
    print(f'{key} finished with time {final_time}s')
    print(f'Time spend per predict: {time_per_iteration}ms')

    print(ovmsclient_performance)
    proto_df = pd.DataFrame(data=ovmsclient_performance, index=[0])
    proto_df.astype('float32')
    proto_df = proto_df.T
    proto_df.columns = ['time_per_iteration[ms]']
    proto_df.to_excel('performance.xlsx', sheet_name='ovmsclient_perf')

if tf_testing:
    tf_performance = {}

    stub = client.prediction_service_stub
    final_time = 0
    for i in range(flow_iterations):
        start_time = datetime.datetime.now()
        proto = tensorflow.make_tensor_proto(np.zeros((1, 3, 224, 224), dtype=np.float32))

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'resnet'
        request.model_spec.version.value = 0
        request.inputs['0'].CopyFrom(proto)

        response = stub.Predict(request, 10.0)

        response_ndarray = tensorflow.make_ndarray(response.outputs['1463'])
        end_time = datetime.datetime.now()
        final_time += (end_time - start_time).total_seconds()
    time_per_iteration = final_time/flow_iterations * 1000
    tf_performance[key] = time_per_iteration
    print(f'{key} finished with time {final_time}s')
    print(f'Time spend per iteration: {time_per_iteration}ms')

    print(tf_performance)
    proto_df = pd.DataFrame(data=tf_performance, index=[0])
    proto_df.astype('float32')
    proto_df = proto_df.T
    proto_df.columns = ['time_per_iteration[ms]']
    proto_df.to_excel('performance.xlsx', sheet_name='tf_perf')
