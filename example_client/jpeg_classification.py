#
# Copyright (c) 2019-2020 Intel Corporation
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
from tensorflow import make_tensor_proto, make_ndarray
import datetime
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import cv2

def crop_resize(img,cropx,cropy):
    y,x,c = img.shape
    if y < cropy:
        img = cv2.resize(img, (x, cropy))
        y = cropy
    if x < cropx:
        img = cv2.resize(img, (cropx,y))
        x = cropx
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

def getJpeg(path, size):
    with open(path, mode='rb') as file:
        content = file.read()

    img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
    # retrived array has BGR format and 0-255 normalization
    # add image preprocessing if needed by the model
    img = crop_resize(img, size, size)
    img = img.astype('float32')
    img = img.transpose(2,0,1).reshape(1,3,size,size)
    print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
    return img

parser = argparse.ArgumentParser(description='Do requests to ie_serving and tf_serving using images in numpy format')
parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1', help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--size',required=False, default=224, type=int, help='The size of the image in the model')
args = vars(parser.parse_args())

channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
input_images = args.get('images_list')
size = args.get('size')
with open(input_images) as f:
    lines = f.readlines()
print('Start processing:')
print('\tModel name: {}'.format(args.get('model_name')))
print('\tImages list file: {}'.format(args.get('images_list')))

i = 0
matched = 0
processing_times = np.zeros((0),int)
imgs = np.zeros((0,3,size, size), np.dtype('<f'))
lbs = np.zeros((0), int)

for line in lines:
    path, label = line.strip().split(" ")
    img = getJpeg(path, size)

    # lbs = np.append(lbs,[int(label)],0)
    # imgs = np.append(imgs, img,0)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.get('model_name')
    request.inputs[args['input_name']].CopyFrom(tf_contrib_util.make_tensor_proto(img, shape=(img.shape)))
    start_time = datetime.datetime.now()
    result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
    end_time = datetime.datetime.now()
    if args['output_name'] not in result.outputs:
        print("Invalid output name", args['output_name'])
        print("Available outputs:")
        for Y in result.outputs:
            print(Y)
        exit(1)
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    output = make_ndarray(result.outputs[args['output_name']])
    nu = np.array(output)
    # for object classification models show imagenet class
    print('Processing time: {:.2f} ms; speed {:.2f} fps'.format(round(duration), 2),
                                                                                  round(1000 / duration, 2)
                                                                                  )
    ma = np.argmax(nu)
    if int(label) == ma:
        matched += 1
    i += 1
    print("Detected:", ma, " Should be:", label)

latency = np.average(processing_times)
accuracy = matched/i

print("Overall accuracy=",accuracy*100,"%")
print("Average latency=",latency,"ms")

# np.save('imgs.npy', imgs)
# np.save('lbs.npy', lbs)
