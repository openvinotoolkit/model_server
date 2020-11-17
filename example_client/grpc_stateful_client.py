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

parser = argparse.ArgumentParser(description='Sends requests via TFS gRPC API using images in numpy format. '
                                             'It displays performance statistics and optionally the model accuracy')
parser.add_argument('--images_numpy_path', required=True, help='numpy in shape [n,w,h,c] or [n,c,h,w]')
parser.add_argument('--labels_numpy_path', required=False, help='numpy in shape [n,1] - can be used to check model accuracy')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1',
                    help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
parser.add_argument('--transpose_input', choices=["False", "True"], default="True",
                    help='Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. default: True',
                    dest="transpose_input")
parser.add_argument('--transpose_method', choices=["nchw2nhwc","nhwc2nchw"], default="nhwc2nchw",
                    help="How the input transposition should be executed: nhwc2nchw or nchw2nhwc",
                    dest="transpose_method")
parser.add_argument('--iterations', default=0,
                    help='Number of requests iterations, as default use number of images in numpy memmap. default: 0 (consume all frames)',
                    dest='iterations', type=int)
# If input numpy file has too few frames according to the value of iterations and the batch size, it will be
# duplicated to match requested number of frames
parser.add_argument('--batchsize', default=1,
                    help='Number of images in a single request. default: 1',
                    dest='batchsize')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--pipeline_name', default='', help='Define pipeline name, must be same as is in service',
                    dest='pipeline_name')
args = vars(parser.parse_args())

channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

processing_times = np.zeros((0),int)

ark_reader = ArchiveReader("rm_lstm4f/test_feat_1_10.ark")

iterations = 0
for key, obj in ark_reader:
    print("Input ark file data range {0}: {1}".format(key, obj.shape))
    iterations += 1

print('Start processing:')
print('\tModel name: {}'.format(args.get('model_name')))
print('\tIterations: {}'.format(iterations))
print('\tImages numpy path: {}'.format(args.get('images_numpy_path')))


iteration = 0
is_pipeline_request = bool(args.get('pipeline_name'))

SEQUENCE_START = 1
SEQUENCE_END = 2

for key, obj in ark_reader:
    batch_size = obj.shape[0]
    print('\tInput name: {}\n'.format(key))
    print('\tInput in shape: {}\n'.format(obj.shape))
    print('\tInput batch size: {}\n'.format(batch_size))
    for x in range(0, batch_size):
        iteration += 1
        request = predict_pb2.PredictRequest()
        #request.model_spec.name = args.get('pipeline_name') if is_pipeline_request else args.get('model_name')
        request.model_spec.name = args.get('model_name')
        # Add 1 dimension

        print('\tTensor before input in shape: {}\n'.format(obj[x].shape))
        print('\tTensor input in shape: {}\n'.format(expand_dims(obj[x], axis=0).shape))

        request.inputs[args['input_name']].CopyFrom(make_tensor_proto(obj[x], shape=(expand_dims(obj[x], axis=0).shape)))
        if iteration == 1:
            #request.inputs['sequence_id'].CopyFrom(make_tensor_proto(123, dtype="uint64"))
            request.inputs['sequence_control_input'].CopyFrom(make_tensor_proto(SEQUENCE_START, dtype="uint32"))
        if iteration == batch_size + 1:
            request.inputs['sequence_control_input'].CopyFrom(make_tensor_proto(SEQUENCE_START, dtype="uint32"))

        start_time = datetime.datetime.now()
        result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
        end_time = datetime.datetime.now()
        if args['output_name'] not in result.outputs:
            print("Invalid output name", args['output_name'])
            print("Available outputs:")
            for Y in result.outputs:
                print(Y)
            exit(1)

        if 'sequence_id' not in result.outputs:
            print("Missing sequence_id in model output")
            print("Available outputs:")
            for Y in result.outputs:
                print(Y)
            exit(1)

        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        output = make_ndarray(result.outputs[args['output_name']])

        nu = np.array(output)
        # for object classification models show imagenet class
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration,round(np.average(duration), 2),
                                                                                  round(1000 * batch_size / np.average(duration), 2)
                                                                                  ))

print_statistics(processing_times, batch_size)
