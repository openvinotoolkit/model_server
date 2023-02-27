#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
sys.path.append("../../../../demos/common/python")

import ast
import grpc
import numpy as np
import classes
import numpy
import datetime
import argparse
from client_utils import print_statistics
from tritonclient.utils import *
import tritonclient.grpc as grpcclient

DataTypeToContentsFieldName = {
    'BOOL' : 'bool_contents',
    'BYTES' : 'bytes_contents',
    'FP32' : 'fp32_contents',
    'FP64' : 'fp64_contents',
    'INT64' : 'int64_contents',
    'INT32' : 'int_contents',
    'UINT64' : 'uint64_contents',
    'UINT32' : 'uint_contents',
    'INT64' : 'int64_contents',
    'INT32' : 'int_contents',
}

def as_numpy(response, name):
    index = 0
    for output in response.outputs:
        if output.name == name:
            shape = []
            for value in output.shape:
                shape.append(value)
            datatype = output.datatype
            field_name = DataTypeToContentsFieldName[datatype]
            contents = getattr(output, "contents")
            contents = getattr(contents, f"{field_name}")
            if index < len(response.raw_output_contents):
                np_array = np.frombuffer(
                    response.raw_output_contents[index], dtype=triton_to_np_dtype(output.datatype))
            elif len(contents) != 0:
                np_array = np.array(contents,
                                    copy=False)
            else:
                np_array = np.empty(0)
            np_array = np_array.reshape(shape)
            return np_array
        else:
            index += 1
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API using images in format supported by OpenCV. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
    parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1',
                        help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
    parser.add_argument('--batchsize', default=1,
                        help='Number of images in a single request. default: 1',
                        dest='batchsize')
    parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                        dest='model_name')
    parser.add_argument('--pipeline_name', default='', help='Define pipeline name, must be same as is in service',
                        dest='pipeline_name')
    parser.add_argument('--input_string', default='', help='Define pipeline name, must be same as is in service',
                        dest='input_string')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
    parser.add_argument('--server_cert', required=False, help='Path to server certificate', default=None)
    parser.add_argument('--client_cert', required=False, help='Path to client certificate', default=None)
    parser.add_argument('--client_key', required=False, help='Path to client key', default=None)


    args = vars(parser.parse_args())

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    processing_times = np.zeros((0),int)
    batch_size = int(args.get('batchsize'))

    print('Start processing:')
    print('\tModel name: {}'.format(args.get('pipeline_name') if bool(args.get('pipeline_name')) else args.get('model_name')))

    iteration = 0
    is_pipeline_request = bool(args.get('pipeline_name'))

    input_string = args.get('input_string')
    inputs_list = input_string.split()

    inputs = []
    inputs.append(grpcclient.InferInput(args['input_name'], [len(inputs_list)], "BYTES"))
    nmpy = numpy.array(inputs_list, dtype=bytes)
    inputs[0].set_data_from_numpy(nmpy)

    outputs = []

    triton_client = grpcclient.InferenceServerClient(
                    url=address,
                    ssl=args['tls'],
                    root_certificates=args['server_cert'],
                    private_key=args['client_key'],
                    certificate_chain=args['client_cert'],
                    verbose=False)

    start_time = datetime.datetime.now()
    results = triton_client.infer(
        model_name= args.get('pipeline_name') if is_pipeline_request else args.get('model_name'),
        inputs=inputs,
        outputs=outputs)

    end_time = datetime.datetime.now()

    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    output = results.as_numpy(args['output_name'])
    print(np.array2string(output))
    # for object classification models show imagenet class
    print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration,round(np.average(duration), 2),
                                                                                round(1000 * len(inputs_list) / np.average(duration), 2)
                                                                                ))
