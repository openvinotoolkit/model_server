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

import sys
sys.path.append("../../common/python")

import numpy as np
import classes
import datetime
import argparse
import tritonclient.grpc as grpcclient
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

def getJpeg(path, size, rgb_image=0):
    with open(path, mode='rb') as file:
        content = file.read()

    img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
    # retrieved array has BGR format and 0-255 normalization
    # format of data is HWC
    # add image preprocessing if needed by the model
    img = crop_resize(img, size, size)
    img = img.astype('float32')
    #convert to RGB instead of BGR if required by model
    if rgb_image:
        img = img[:, :, [2, 1, 0]]
    # switch from HWC to CHW and reshape to 1,3,size,size for model blob input requirements
    img = img.transpose(2,0,1).reshape(1,3,size,size)
    print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
    return img

parser = argparse.ArgumentParser(description='Do requests to OpenVINO Model Server using images in numpy format')
parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1', help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                    dest='model_name')
parser.add_argument('--size',required=False, default=224, type=int, help='The size of the image in the model')
parser.add_argument('--rgb_image',required=False, default=0, type=int, help='Convert BGR channels to RGB channels in the input image')
args = vars(parser.parse_args())

client = grpcclient.InferenceServerClient(url="{}:{}".format(args['grpc_address'],args['grpc_port']))
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

rgb_image = args.get('rgb_image')
for line in lines:
    path, label = line.strip().split(" ")
    img = getJpeg(path, size, rgb_image)

    infer_input = grpcclient.InferInput(args['input_name'], img.shape, "FP32")
    infer_input.set_data_from_numpy(img)
    start_time = datetime.datetime.now()
    result = client.infer(args.get('model_name'), [infer_input])
    end_time = datetime.datetime.now()
    if args['output_name'] not in [output.name for output in result.get_output()]:
        print("Invalid output name", args['output_name'])
        print("Available outputs:")
        for out in result.get_output():
            print(out.name)
        exit(1)
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    output = result.as_numpy(args['output_name'])
    nu = np.array(output)
    # for object classification models show imagenet class
    print('Processing time: {:.2f} ms; speed {:.2f} fps'.format(round(duration, 2), round(1000 / duration, 2)))
    offset = 0
    if nu.shape[1] == 1001:
        offset = 1 
    ma = np.argmax(nu) - offset
    mark_message = ""
    if int(label) == ma:
        matched += 1
        mark_message = "; Correct match."
    else:
        mark_message = "; Incorrect match. Should be {} {}".format(label, classes.imagenet_classes[int(label)])
    i += 1
    print("\t",i, classes.imagenet_classes[ma],ma, mark_message)

latency = np.average(processing_times)
accuracy = matched/i

print("Overall accuracy=",accuracy*100,"%")
print("Average latency=",latency,"ms")

