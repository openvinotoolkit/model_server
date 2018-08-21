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

from grpc.beta import implementations
import numpy as np
import tensorflow.contrib.util as tf_contrib_util
import classes
import datetime
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2



parser = argparse.ArgumentParser(description='Do requests to ie_serving and tf_serving using images in numpy format')
parser.add_argument('--images_numpy_path', required=True, help='numpy in shape [n,w,h,c]')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service')
parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name')
parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1', help='Specify output name')
parser.add_argument('--transpose_input',required=False, default=True, help='Set to False to skip NHWC->NCHW input transposing')

args = vars(parser.parse_args())

channel = implementations.insecure_channel(args['grpc_address'], int(args['grpc_port']))

stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
processing_times = np.zeros((0),int)
imgs = np.load(args['images_numpy_path'], mmap_mode='r', allow_pickle=False)

if args['transpose_input']:
    imgs = imgs.transpose((0,3,1,2))
for x in range(100):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    img = imgs[x,:,:,:]
    request.inputs[args['input_name']].CopyFrom(tf_contrib_util.make_tensor_proto(img, shape=list((1,)+img.shape)))
    start_time = datetime.datetime.now()
    result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    output = tf_contrib_util.make_ndarray(result.outputs[args['output_name']])

    nu = np.array(output)
    ma = np.argmax(nu)
    # for object classification models show imagenet class
    print("Best classification", classes.imagenet_classes[ma])

print('processing time for all iterations')
for x in processing_times:
    print(x, 'ms')
print('processing_statistics')
print('average time:',round(np.average(processing_times),1), 'ms; average speed:', round(1000/np.average(processing_times),1),'fps')
print('median time:',round(np.median(processing_times),1), 'ms; median speed:',round(1000/np.median(processing_times),1),'fps')
print('max time:',round(np.max(processing_times),1), 'ms; max speed:',round(1000/np.max(processing_times),1),'fps')
print('min time:',round(np.min(processing_times),1),'ms; min speed:',round(1000/np.min(processing_times),1),'fps')
print('time percentile 90:',round(np.percentile(processing_times,90),1),'ms; speed percentile 90:',round(1000/np.percentile(processing_times,90),1),'fps')
print('time percentile 50:',round(np.percentile(processing_times,50),1),'ms; speed percentile 50:',round(1000/np.percentile(processing_times,50),1),'fps')
print('time standard deviation:',round(np.std(processing_times)))
print('time variance:',round(np.var(processing_times)))
