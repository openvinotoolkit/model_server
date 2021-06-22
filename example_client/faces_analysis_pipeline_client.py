#
# Copyright (c) 2021 Intel Corporation
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
import cv2
import os
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray
import argparse
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
parser = argparse.ArgumentParser(description='Client for detailed faces analysis pipeline')
parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9178, help='Specify port to grpc service. default: 9178')
parser.add_argument('--pipeline_name', required=False, default='ocr', help='Pipeline name to request. default: ocr')
parser.add_argument('--image_input_name', required=False, default='image', help='Pipeline input name for input with image with faces. default: image')
parser.add_argument('--image_input_path', required=True, help='Input image path.')
parser.add_argument('--face_images_output_name', required=False, default='face_images', help='Pipeline output name for cropped images with faces. default: face_images')
parser.add_argument('--face_images_save_path', required=False, default='', help='If specified, face images will be saved to disk.')
parser.add_argument('--image_width', required=False, default=600, help='Pipeline input image width. default: 600')
parser.add_argument('--image_height', required=False, default=400, help='Pipeline input image height. default: 400')
parser.add_argument('--input_image_layout', required=False, default='NHWC', choices=['NCHW', 'NHWC', 'BINARY'], help='Pipeline input image layout. default: NHWC')

args = vars(parser.parse_args())

def prepare_img_input_in_nchw_format(request, name, path, resize_to_shape):
    img = cv2.imread(path).astype(np.float32)  # BGR color format, shape HWC
    img = cv2.resize(img, (resize_to_shape[1], resize_to_shape[0]))
    target_shape = (img.shape[0], img.shape[1])
    img = img.transpose(2,0,1).reshape(1,3,target_shape[0],target_shape[1])
    request.inputs[name].CopyFrom(make_tensor_proto(img, shape=img.shape))

def prepare_img_input_in_nhwc_format(request, name, path, resize_to_shape):
    img = cv2.imread(path).astype(np.float32)  # BGR color format, shape HWC
    img = cv2.resize(img, (resize_to_shape[1], resize_to_shape[0]))
    target_shape = (img.shape[0], img.shape[1])
    img = img.reshape(1,target_shape[0],target_shape[1],3)
    request.inputs[name].CopyFrom(make_tensor_proto(img, shape=img.shape))

def prepare_img_input_in_binary_format(request, name, path):
    with open(path, 'rb') as f:
        data = f.read()
        request.inputs[name].CopyFrom(make_tensor_proto(data, shape=[1]))

def save_face_images_as_jpgs(output_nd, name, location):
    for i in range(output_nd.shape[0]):
        out = output_nd[i][0]
        if len(out.shape) == 3 and out.shape[0] == 3:  # NCHW
            out = out.transpose(1,2,0)
        cv2.imwrite(os.path.join(location, name + '_' + str(i) + '.jpg'), out)

def update_people_ages(output_nd, people):
    for i in range(output_nd.shape[0]):
        age = int(output_nd[i,0,0,0,0] * 100)
        if len(people) < i + 1:
            people.append({'age': age})
        else:
            people[i].update({'age': age})
    return people

def update_people_genders(output_nd, people):
    for i in range(output_nd.shape[0]):
        gender = 'male' if output_nd[i,0,0,0,0] < output_nd[i,0,1,0,0] else 'female'
        if len(people) < i + 1:
            people.append({'gender': gender})
        else:
            people[i].update({'gender': gender})
    return people

def update_people_emotions(output_nd, people):
    emotion_names = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'surprised',
        4: 'angry'
    }
    for i in range(output_nd.shape[0]):
        emotion_id = np.argmax(output_nd[i,0,:,0,0])
        emotion = emotion_names[emotion_id]
        if len(people) < i + 1:
            people.append({'emotion': emotion})
        else:
            people[i].update({'emotion': emotion})
    return people

def update_people_coordinate(output_nd, people):
    for i in range(output_nd.shape[0]):
        if len(people) < i + 1:
            people.append({'coordinate': output_nd[i,0,:]})
        else:
            people[i].update({'coordinate': output_nd[i,0,:]})
    return people


address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
channel = grpc.insecure_channel(address,
    options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = args['pipeline_name']

if args['input_image_layout'] == 'NCHW':
    prepare_img_input_in_nchw_format(request, args['image_input_name'], args['image_input_path'], (int(args['image_height']), int(args['image_width'])))
elif args['input_image_layout'] == 'NHWC':
    prepare_img_input_in_nhwc_format(request, args['image_input_name'], args['image_input_path'], (int(args['image_height']), int(args['image_width'])))
else:
    prepare_img_input_in_binary_format(request, args['image_input_name'], args['image_input_path'])

try:
    response = stub.Predict(request, 30.0)
except grpc.RpcError as err:
    if err.code() == grpc.StatusCode.ABORTED:
        print('No face has been found in the image')
        exit(1)
    else:
        raise err

people = []

for name in response.outputs:
    print(f"Output: name[{name}]")
    tensor_proto = response.outputs[name]
    output_nd = make_ndarray(tensor_proto)
    print(f"    numpy => shape[{output_nd.shape}] data[{output_nd.dtype}]")

    if name == args['face_images_output_name'] and len(args['face_images_save_path']) > 0:
        save_face_images_as_jpgs(output_nd, name, args['face_images_save_path'])

    if name == 'ages':
        people = update_people_ages(output_nd, people)
    if name == 'genders':
        people = update_people_genders(output_nd, people)
    if name == 'emotions':
        people = update_people_emotions(output_nd, people)
    if name == 'face_coordinates':
        people = update_people_coordinate(output_nd, people)


print('\nFound', len(people), 'faces:')
for person in people:
    print('Age:', person['age'], '; Gender:', person['gender'], '; Emotion:', person['emotion'], '; Original image coordinate:', person['coordinate'])
