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

import grpc
import numpy as np
import classes
from enum import Enum, auto
import datetime
import argparse
from client_utils import print_statistics
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import *

class DataType(Enum):
    INVALID = auto()
    BOOL = auto()
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    FP16 = auto()
    FP32 = auto()
    FP64 = auto()
    STRING = auto()

def _triton_datatype_to_contents_field_name(datatype):
    """
        FIELD_NAME:         DATATYPE:
        bool_contents       "BOOL"
        bytes_contents      "BYTES"
        fp32_contents       "FP32"
        fp64_contents       "FP64"
        int64_contents      "INT64"
        int_contents        "INT32"
        uint64_contents     "UINT64"
        uint_contents       "UINT32"
    """
    if datatype == DataType.INT32.name:
        return "int_contents"
    elif datatype == DataType.UINT32.name:
        return "uint_contents"
    else:
        return f"{datatype.lower()}_contents"    # all other types

def as_numpy(response, name):
    index = 0
    for output in response.outputs:
        if output.name == name:
            shape = []
            for value in output.shape:
                shape.append(value)
            datatype = output.datatype
            field_name = _triton_datatype_to_contents_field_name(datatype)
            contents = eval(f"output.contents.{field_name}[:]")
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
    parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
    parser.add_argument('--labels_numpy_path', required=False, help='numpy in shape [n,1] - can be used to check model accuracy')
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

    args = vars(parser.parse_args())

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    processing_times = np.zeros((0),int)

    input_images = args.get('images_list')
    with open(input_images) as f:
        lines = f.readlines()
    batch_size = int(args.get('batchsize'))
    while batch_size > len(lines):
        lines += lines

    if args.get('labels_numpy_path') is not None:
        lbs = np.load(args['labels_numpy_path'], mmap_mode='r', allow_pickle=False)
        matched_count = 0
        total_executed = 0
    batch_size = int(args.get('batchsize'))

    print('Start processing:')
    print('\tModel name: {}'.format(args.get('pipeline_name') if bool(args.get('pipeline_name')) else args.get('model_name')))

    iteration = 0
    is_pipeline_request = bool(args.get('pipeline_name'))

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(address)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    batch_i = 0
    image_data = []
    labels = []
    for line in lines:
        batch_i += 1
        path, label = line.strip().split(" ")
        with open(path, 'rb') as f:
            image_data.append(f.read())
        labels.append(label)
        if batch_i < batch_size:
            continue
        inputs = []
        inputs.append(service_pb2.ModelInferRequest().InferInputTensor())
        inputs[0].name = args['input_name']
        inputs[0].datatype = "BYTES"
        inputs[0].shape.extend([1])
        inputs[0].contents.bytes_contents.append(image_data[0])

        outputs = []
        outputs.append(service_pb2.ModelInferRequest().InferRequestedOutputTensor())
        outputs[0].name = "prob"

        request = service_pb2.ModelInferRequest()
        request.model_name = args.get('pipeline_name') if is_pipeline_request else args.get('model_name')
        request.inputs.extend(inputs)

        start_time = datetime.datetime.now()
        request.outputs.extend(outputs)
        response = grpc_stub.ModelInfer(request)
        end_time = datetime.datetime.now()

        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        output = as_numpy(response, args['output_name'])
        nu = np.array(output)
        # for object classification models show imagenet class
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration,round(np.average(duration), 2),
                                                                                    round(1000 * batch_size / np.average(duration), 2)
                                                                                    ))
        # Comment out this section for non imagenet datasets
        print("imagenet top results in a single batch:")
        for i in range(nu.shape[0]):
            lbs_i = iteration * batch_size
            single_result = nu[[i],...]
            offset = 0
            if nu.shape[1] == 1001:
                offset = 1
            ma = np.argmax(single_result) - offset
            mark_message = ""
            if args.get('labels_numpy_path') is not None:
                total_executed += 1
                if ma == lbs[lbs_i + i]:
                    matched_count += 1
                    mark_message = "; Correct match."
                else:
                    mark_message = "; Incorrect match. Should be {} {}".format(lbs[lbs_i + i], classes.imagenet_classes[lbs[lbs_i + i]] )
            print("\t",i, classes.imagenet_classes[ma],ma, mark_message)

        # Comment out this section for non imagenet datasets
        iteration += 1
        image_data = []
        labels = []
        batch_i = 0

    print_statistics(processing_times, batch_size)

    if args.get('labels_numpy_path') is not None:
        print('Classification accuracy: {:.2f}'.format(100*matched_count/total_executed))

