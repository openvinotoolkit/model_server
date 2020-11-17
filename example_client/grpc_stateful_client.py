#
# Copyright (c) 2018-2020 Intel Corporation
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

import grpc
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray, expand_dims
import classes
import datetime
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import print_statistics, prepare_certs
from kaldi_python_io import ArchiveReader

parser = argparse.ArgumentParser(description='Sends requests via TFS gRPC API using data in stateful model ark input file. '
                                             'It displays performance statistics and optionally')
parser.add_argument('--model_input_path', required=False, default='rm_lstm4f/test_feat_1_10.ark' help='path to input ark file')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='Parameter', help='Specify input tensor name. default: Parameter')
parser.add_argument('--output_name',required=False, default='affinetransform/Fused_Add_',
                    help='Specify output name. default: affinetransform/Fused_Add_')
parser.add_argument('--model_name', default='rm_lstm4f', help='Define model name, must be same as is in service. default: rm_lstm4f',
                    dest='model_name')

args = vars(parser.parse_args())

channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

processing_times = np.zeros((0),int)

model_input_path = args.get('model_input_path')
print('Reading ark file {}'.format(model_input_path)
ark_reader = ArchiveReader("model_input_path")

for key, obj in ark_reader:
    print("Input ark file data range {0}: {1}".format(key, obj.shape))

print('Start processing:')
print('\tModel name: {}'.format(args.get('model_name')))

SEQUENCE_START = 1
SEQUENCE_END = 2

sequence_id = 4
for key, obj in ark_reader:
    batch_size = obj.shape[0]
    print('\tInput name: {}\n'.format(key))
    print('\tInput in shape: {}\n'.format(obj.shape))
    print('\tInput batch size: {}\n'.format(batch_size))
    print('\tSequence id: {}\n'.format(sequence_id))
    for x in range(0, batch_size):

        print('\tExecution: {}\n'.format(x))
        request = predict_pb2.PredictRequest()
        request.model_spec.name = args.get('model_name')

        print('\tTensor before input in shape: {}\n'.format(obj[x].shape))
        print('\tTensor input in shape: {}\n'.format(expand_dims(obj[x], axis=0).shape))

        request.inputs[args['input_name']].CopyFrom(make_tensor_proto(obj[x], shape=(expand_dims(obj[x], axis=0).shape)))
        if x == 0:
            request.inputs['sequence_control_input'].CopyFrom(make_tensor_proto(SEQUENCE_START, dtype="uint32"))
        
        request.inputs['sequence_id'].CopyFrom(make_tensor_proto(sequence_id, dtype="uint64"))

        if x == batch_size:
            request.inputs['sequence_control_input'].CopyFrom(make_tensor_proto(SEQUENCE_END, dtype="uint32"))

        start_time = datetime.datetime.now()
        result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
        end_time = datetime.datetime.now()
        if args['output_name'] not in result.outputs:
            print("ERROR: Invalid output name", args['output_name'])
            print("Available outputs:")
            for Y in result.outputs:
                print(Y)
            exit(1)

        if 'sequence_id' not in result.outputs:
            print("ERROR: Missing sequence_id in model output")
            print("Available outputs:")
            for Y in result.outputs:
                print(Y)
            exit(1)

        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        output = make_ndarray(result.outputs[args['output_name']])

        nu = np.array(output)
        # for object classification models show imagenet class
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(x,round(np.average(duration), 2),
                                                                                  round(1000 * batch_size / np.average(duration), 2)
                                                                                  ))
    sequence_id += 1

print_statistics(processing_times, batch_size)
