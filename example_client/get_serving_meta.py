#
# Copyright (c) 2018 Intel Corporation
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

import argparse
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

def model_metadata_response(response):
    signature_def = response.metadata['signature_def']
    signature_map = get_model_metadata_pb2.SignatureDefMap()
    signature_map.ParseFromString(signature_def.value)
    serving_default = signature_map.ListFields()[0][1]['serving_default']
    serving_inputs = serving_default.inputs
    input_blobs_keys = {key: {} for key in serving_inputs.keys()}
    tensor_shape = {key: serving_inputs[key].tensor_shape
                    for key in serving_inputs.keys()}
    for input_blob in input_blobs_keys:
        inputs_shape = [d.size for d in tensor_shape[input_blob].dim]
        tensor_dtype = serving_inputs[input_blob].dtype
        input_blobs_keys[input_blob].update({'shape': inputs_shape})
        input_blobs_keys[input_blob].update({'dtype': tensor_dtype})

    serving_outputs = serving_default.outputs
    output_blobs_keys = {key: {} for key in serving_outputs.keys()}
    tensor_shape = {key: serving_outputs[key].tensor_shape
                    for key in serving_outputs.keys()}
    for output_blob in output_blobs_keys:
        outputs_shape = [d.size for d in tensor_shape[output_blob].dim]
        tensor_dtype = serving_outputs[output_blob].dtype
        output_blobs_keys[output_blob].update({'shape': outputs_shape})
        output_blobs_keys[output_blob].update({'dtype': tensor_dtype})

    return input_blobs_keys, output_blobs_keys

dtype_name = [ 'DT_INVALID',
               'DT_FLOAT',
               'DT_DOUBLE',
               'DT_INT32',
               'DT_UINT8',
               'DT_INT16',
               'DT_INT8',
               'DT_STRING',
               'DT_COMPLEX64',
               'DT_INT64',
               'DT_BOOL',
               'DT_QINT8',
               'DT_QUINT8',
               'DT_QINT32',
               'DT_BFLOAT16',
               'DT_QINT16',
               'DT_QUINT16',
               'DT_UINT16',
               'DT_COMPLEX128',
               'DT_HALF',
               'DT_RESOURCE',
               'DT_VARIANT',
               'DT_UINT32',
               'DT_UINT64']
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto

parser = argparse.ArgumentParser(description='Get information about served models')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=None, type=int, help='Define model version - must be numerical',
                    dest='model_version')
args = vars(parser.parse_args())

channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

print('Getting model metadata for model:',args.get('model_name'))

metadata_field = "signature_def"
request = get_model_metadata_pb2.GetModelMetadataRequest()
request.model_spec.name = args.get('model_name')
if args.get('model_version') is not None:
    request.model_spec.version.value = args.get('model_version')
request.metadata_field.append(metadata_field)

result = stub.GetModelMetadata(request, 10.0) # result includes a dictionary with all model outputs
input_metadata, output_metadata = model_metadata_response(
    response=result)
print('Inputs metadata:')
for i in input_metadata:
    print("\tInput name: {}; shape: {}; dtype: {}"
          .format(i, input_metadata[i]['shape'],
                  dtype_name[input_metadata[i]['dtype']]))
print('Outputs metadata:')
for i in output_metadata:
    print("\tOutput name: {}; shape: {}; dtype: {}"
          .format(i, output_metadata[i]['shape'],
                  dtype_name[output_metadata[i]['dtype']]))


