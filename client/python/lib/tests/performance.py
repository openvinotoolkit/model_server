from ovmsclient.tfs_compat.grpc.tensors import make_tensor_proto, make_ndarray
# from tensorflow import make_tensor_proto, make_ndarray
import datetime
import numpy as np
from tensorflow.core.framework.types_pb2 import DataType
import pandas as pd

np.random.seed(0)

# parameters
iterations = 100_000
lower_iterations = 10_000
make_tensor_proto_testing = False
make_ndarray_testing = True
proto_inputs_dict = {
    # "float_shape_dtype_no_reshape": {
    #     "values": 5.0,
    #     "shape": [1],
    #     "dtype": DataType.DT_FLOAT
    # },
    # "float_no_shape_dtype_no_reshape": {
    #     "values": 5.0,
    #     "dtype": DataType.DT_FLOAT
    # },
    # "bytes_shape_dtype_no_reshape": {
    #     "values": bytes([1, 2, 3]),
    #     "shape": [1],
    #     "dtype": DataType.DT_STRING
    # },
    # "bytes_no_shape_dtype_no_reshape": {
    #     "values": bytes([1, 2, 3]),
    #     "dtype": DataType.DT_STRING
    # },
    # "list_shape_dtype_no_reshape": {
    #     "values": [1, 2, 3],
    #     "shape": [3],
    #     "dtype": DataType.DT_INT64
    # },
    # "list_no_shape_dtype_no_reshape": {
    #     "values": [1, 2, 3],
    #     "dtype": DataType.DT_INT64
    # },
    # "2Dlist_shape_dtype_no_reshape": {
    #     "values": [[1, 2, 3], [1, 2, 3]],
    #     "shape": [2,3],
    #     "dtype": DataType.DT_INT64
    # },
    # "2Dlist_no_shape_dtype_no_reshape": {
    #     "values": [[1, 2, 3], [1, 2, 3]],
    #     "dtype": DataType.DT_INT64
    # },
    # "binary_shape_dtype_no_reshape": {
    #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    #     "shape": [2],
    #     "dtype": DataType.DT_STRING
    # },
    # "binary_no_shape_dtype_no_reshape": {
    #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    #     "dtype": DataType.DT_STRING
    # },
    # # "binary_to_dtype_shape_dtype_no_reshape": {   # need to be comment for tensorflow
    # #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    # #     "shape": [6],
    # #     "dtype": DataType.DT_INT8
    # # },
    # # "binary_to_dtype_no_shape_dtype_no_reshape": {    # need to be comment for tensorflow
    # #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    # #     "dtype": DataType.DT_INT8
    # # },
    # "ndarray_shape_dtype_no_reshape": {
    #     "values": np.array([1, 2, 3]),
    #     "shape": [3],
    #     "dtype": DataType.DT_FLOAT
    # },
    # "ndarray_no_shape_dtype_no_reshape": {
    #     "values": np.array([1, 2, 3]),
    #     "dtype": DataType.DT_FLOAT
    # },
    # "2Dndarray_shape_dtype_no_reshape": {
    #     "values": np.array([[1, 2, 3], [1, 2, 3]]),
    #     "shape": [2,3],
    #     "dtype": DataType.DT_FLOAT
    # },
    # "2Dndarray_no_shape_dtype_no_reshape": {
    #     "values": np.array([[1, 2, 3], [1, 2, 3]]),
    #     "dtype": DataType.DT_FLOAT
    # },
    # "float_shape_no_dtype_no_reshape": {
    #     "values": 5.0,
    #     "shape": [1],
    # },
    # "float_no_shape_no_dtype_no_reshape": {
    #     "values": 5.0,
    # },
    # "bytes_shape_no_dtype_no_reshape": {
    #     "values": bytes([1, 2, 3]),
    #     "shape": [1],
    # },
    # "bytes_no_shape_no_dtype_no_reshape": {
    #     "values": bytes([1, 2, 3]),
    # },
    # "list_shape_no_dtype_no_reshape": {
    #     "values": [1, 2, 3],
    #     "shape": [3]
    # },
    # "list_no_shape_no_dtype_no_reshape": {
    #     "values": [1, 2, 3],
    # },
    # "2Dlist_shape_no_dtype_no_reshape": {
    #     "values": [[1, 2, 3], [1, 2, 3]],
    #     "shape": [2,3],
    # },
    # "2Dlist_no_shape_no_dtype_no_reshape": {
    #     "values": [[1, 2, 3], [1, 2, 3]],
    # },
    # "binary_shape_no_dtype_no_reshape": {
    #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    #     "shape": [2],
    # },
    # "binary_no_shape_no_dtype_no_reshape": {
    #     "values": [bytes([1, 2, 3]), bytes([1, 2, 3])],
    # },
    # # "binary_to_dtype_shape_no_dtype_no_reshape": {    #impossible
    # # },
    # # "binary_to_dtype_no_shape_no_dtype_no_reshape": { #impossible
    # # },
    # "ndarray_shape_no_dtype_no_reshape": {
    #     "values": np.array([1.0, 2.0, 3.0]),
    #     "shape": [3],
    # },
    # "ndarray_no_shape_no_dtype_no_reshape": {
    #     "values": np.array([1.0, 2.0, 3.0]),
    # },
    # "2Dndarray_shape_no_dtype_no_reshape": {
    #     "values": np.array([[1, 2, 3], [1, 2, 3]]),
    #     "shape": [2,3],
    # },
    # "2Dndarray_no_shape_no_dtype_no_reshape": {
    #     "values": np.array([[1, 2, 3], [1, 2, 3]]),
    # },
    # # "float_shape_dtype_reshape": {   #impossible
    # # },
    # # "float_no_shape_dtype_reshape": {    #impossible
    # # },
    # # "list_shape_dtype_reshape": { #impossible
    # # },
    # # "list_no_shape_dtype_reshape": {  #impossible
    # # },
    # "2Dlist_shape_dtype_reshape": {
    #     "values": [[1, 2, 3], [1, 2, 3]],
    #     "shape": [6],
    #     "dtype": DataType.DT_INT64
    # },
    # # "2Dlist_no_shape_dtype_reshape": {    #impossible
    # # },
    # # "binary_shape_dtype_reshape": {   #impossible
    # # },
    # # "binary_no_shape_dtype_reshape": {    #impossible
    # # },
    # # "ndarray_shape_dtype_reshape": {  #impossible
    # # },
    # # "ndarray_no_shape_dtype_reshape": {   #impossible
    # # },
    # "2Dndarray_shape_dtype_reshape": {
    #     "values": np.array([[1, 2, 3], [1, 2, 3]]),
    #     "shape": [6],
    #     "dtype": DataType.DT_FLOAT
    # },
    # # "2Dndarray_no_shape_no_dtype_reshape": {   #impossible
    # # },
    # # "scalar_shape_no_dtype_reshape": {    #impossible
    # # },
    # # "scalar_no_shape_no_dtype_reshape": { #impossible
    # # },
    # # "bytes_shape_no_dtype_reshape": { #impossible
    # # },
    # # "bytes_no_shape_no_dtype_reshape": {  #impossible
    # # },
    # # "list_shape_no_dtype_reshape": {  #impossible
    # # },
    # # "list_no_shape_no_dtype_reshape": {   #impossible
    # # },
    # "2Dlist_shape_no_dtype_reshape": {
    #     "values": [[1, 2, 3], [1, 2, 3]],
    #     "shape": [6],
    # },
    # # "2Dlist_no_shape_no_dtype_reshape": {   #impossible
    # # },
    # # "binary_shape_no_dtype_reshape": {    #impossible
    # # },
    # # "binary_no_shape_no_dtype_reshape": { #impossible
    # # },
    # # "binary_to_dtype_shape_no_dtype_reshape": {   #impossible
    # # },
    # # "binary_to_dtype_no_shape_no_dtype_reshape": {    #impossible
    # # },
    # # "ndarray_shape_no_dtype_reshape": {   #impossible
    # # },
    # # "ndarray_no_shape_no_dtype_reshape": {    #impossible
    # # },
    # "2Dndarray_shape_no_dtype_reshape": {
    #     "values": np.array([[1, 2, 3], [1, 2, 3]]),
    #     "shape": [6],
    # },
    # # "2Dndarray_no_shape_no_dtype_reshape": {    #impossible
    # # },
    # "ndarray_zeros_resnet_npdtype_dtype_no_reshape": {
    #     "values": np.zeros((1,3,244,244), dtype=np.float32),
    #     "dtype": DataType.DT_FLOAT
    # },
    # "ndarray_zeros_resnet_no_npdtype_dtype_no_reshape": {
    #     "values": np.zeros((1,3,244,244)),
    #     "dtype": DataType.DT_INT16
    # },
    # "ndarray_zeros_resnet_npdtype_no_dtype_no_reshape": {
    #     "values": np.zeros((1,3,244,244), dtype=np.float32),
    # },
    # "ndarray_zeros_resnet_no_npdtype_no_dtype_no_reshape": {
    #     "values": np.zeros((1,3,244,244)),
    # },
    # "ndarray_zeros_resnet_npdtype_dtype_reshape": {
    #     "values": np.zeros((1,3,244,244), dtype=np.float32),
    #     "dtype": DataType.DT_FLOAT,
    #     "shape": [1,2,24,3721],
    # },
    # "ndarray_zeros_resnet_no_npdtype_dtype_reshape": {
    #     "values": np.zeros((1,3,244,244)),
    #     "dtype": DataType.DT_INT16,
    #     "shape": [1,2,24,3721],
    # },
    # "ndarray_zeros_resnet_npdtype_no_dtype_reshape": {
    #     "values": np.zeros((1,3,244,244), dtype=np.float32),
    #     "shape": [1,2,24,3721],
    # },
    # "ndarray_zeros_resnet_no_npdtype_no_dtype_reshape": {
    #     "values": np.zeros((1,3,244,244)),
    #     "shape": [1,2,24,3721],
    # },
    # "ndarray_rand_resnet_npdtype_dtype": {
    #     "values": np.random.rand(1,3,244,244).astype(np.float32),
    #     "dtype": DataType.DT_FLOAT,
    # },
    # "ndarray_rand_resnet_no_npdtype_dtype": {
    #     "values": np.random.rand(1,3,244,244),
    #     "dtype": DataType.DT_FLOAT,
    # },
    # "ndarray_rand_resnet_npdtype_no_dtype": {
    #     "values": np.random.rand(1,3,244,244).astype(np.float32),
    # },
    # "ndarray_rand_resnet_no_npdtype_no_dtype": {
    #     "values": np.random.rand(1,3,244,244),
    # },
    "Big_2Dlist_shape_dtype_no_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
        "shape": [3,2,5000],
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
        "shape": [3,2,5000],
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
        "shape": [2,6,2500],
        "dtype": DataType.DT_INT64
    },
    # "Big_2Dlist_no_shape_dtype_reshape": {    #impossible
    # },
    "Big_2Dlist_shape_no_dtype_reshape": {
        "values": [[[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]],
                   [[i for i in range(5000)], [i for i in range(5000)]]],
        "shape": [2,6,2500],
    },
    # "Big_2Dlist_no_shape_no_dtype_reshape": {   #impossible
    # },
    # "Big_binary_shape_dtype_no_reshape": {
    #     "values": [bytes([i%256 for i in range(10000)]),
    #                bytes([(3*i)%256 for i in range(10000)]),
    #                bytes([(7*i)%256 for i in range(10000)])],
    #     "shape": [3],
    #     "dtype": DataType.DT_STRING
    # },
    # "Big_binary_no_shape_dtype_no_reshape": {
    #     "values": [bytes([i%256 for i in range(10000)]),
    #                bytes([(3*i)%256 for i in range(10000)]),
    #                bytes([(7*i)%256 for i in range(10000)])],
    #     "dtype": DataType.DT_STRING
    # },
    # "Big_binary_shape_no_dtype_no_reshape": {
    #     "values": [bytes([i%256 for i in range(10000)]),
    #                bytes([(3*i)%256 for i in range(10000)]),
    #                bytes([(7*i)%256 for i in range(10000)])],
    #     "shape": [3],
    # },
    # "Big_binary_no_shape_no_dtype_no_reshape": {
    #     "values": [bytes([i%256 for i in range(10000)]),
    #                bytes([(3*i)%256 for i in range(10000)]),
    #                bytes([(7*i)%256 for i in range(10000)])],
    # },
}

lower_iteration_cases = [
    "Big_2Dlist_shape_dtype_no_reshape", "Big_2Dlist_no_shape_dtype_no_reshape",
    "Big_2Dlist_shape_no_dtype_no_reshape", "Big_2Dlist_no_shape_no_dtype_no_reshape",
    "Big_2Dlist_shape_dtype_reshape", "Big_2Dlist_shape_no_dtype_reshape"
]

# make_tensor_proto performance testing
if(make_tensor_proto_testing):
    make_tensor_proto_performance = {}

    for key, value in proto_inputs_dict.items():
        final_time = 0
        iteration = iterations if key not in lower_iteration_cases else lower_iterations
        for i in range(iteration):
            start_time = datetime.datetime.now()
            proto = make_tensor_proto(**value)
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
if(make_ndarray_testing):
    make_ndarray_performance = {}

    for key, value in proto_inputs_dict.items():
        final_time = 0
        iteration = iterations if key not in lower_iteration_cases else lower_iterations
        for i in range(iteration):
            proto = make_tensor_proto(**value)
            print(proto.dtype)
            exit()
            start_time = datetime.datetime.now()
            array = make_ndarray(proto)
            end_time = datetime.datetime.now()
            final_time += (end_time - start_time).total_seconds()
        time_per_ndarray = final_time/iteration * 1000
        make_ndarray_performance[key] = time_per_ndarray
        print(f'{key} finished with time {final_time}s')
        print(f'Time spend per proto: {time_per_ndarray}ms')

    print(make_ndarray_performance)
    ndarray_df = pd.DataFrame(data=make_ndarray_performance, index=[0])
    ndarray_df.astype('float32')
    ndarray_df = ndarray_df.T
    ndarray_df.columns = ['time_per_ndarray[ms]']
    ndarray_df.to_excel('performance.xlsx', sheet_name='make_ndarray_perf')
