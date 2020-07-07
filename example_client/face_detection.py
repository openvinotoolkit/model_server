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

import argparse
import cv2
import datetime
import grpc
import numpy as np
import os
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from client_utils import print_statistics


def load_image(file_path):
    img = cv2.imread(file_path)  # BGR color format, shape HWC
    img = cv2.resize(img, (args['width'], args['height']))
    img = img.transpose(2,0,1).reshape(1,3,args['height'],args['width'])
    # change shape to NCHW
    return img


parser = argparse.ArgumentParser(description='Demo for object detection requests via TFS gRPC API.'
                                             'analyses input images and saveswith with detected objects.'
                                             'it relies on model given as parameter...')

parser.add_argument('--model_name', required=False, help='Name of the model to be used', default="face-detection")
parser.add_argument('--input_images_dir', required=False, help='Directory with input images', default="images/people")
parser.add_argument('--output_dir', required=False, help='Directory for staring images with detection results', default="results")
parser.add_argument('--batch_size', required=False, help='How many images should be grouped in one batch', default=1, type=int)
parser.add_argument('--width', required=False, help='How the input image width should be resized in pixels', default=1200, type=int)
parser.add_argument('--height', required=False, help='How the input image width should be resized in pixels', default=800, type=int)
parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')

args = vars(parser.parse_args())

channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'],args['grpc_port']))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

files = os.listdir(args['input_images_dir'])
batch_size = args['batch_size']
model_name = args['model_name']
print("Running "+model_name+" on files:" + str(files))

imgs = np.zeros((0,3,args['height'],args['width']), np.dtype('<f'))
for i in files:
    img = load_image(os.path.join(args['input_images_dir'], i))
    imgs = np.append(imgs, img, axis=0)  # contains all imported images

print('Start processing {} iterations with batch size {}'.format(len(files)//batch_size , batch_size))

iteration = 0
processing_times = np.zeros((0),int)


for x in range(0, imgs.shape[0] - batch_size + 1, batch_size):
    iteration += 1
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    img = imgs[x:(x + batch_size)]
    print("\nRequest shape", img.shape)
    request.inputs["data"].CopyFrom(make_tensor_proto(img, shape=(img.shape)))
    start_time = datetime.datetime.now()
    result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
    end_time = datetime.datetime.now()

    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    output = make_ndarray(result.outputs["detection_out"])
    print("Response shape", output.shape)
    for y in range(0,img.shape[0]):  # iterate over responses from all images in the batch
        img_out = img[y,:,:,:]

        print("image in batch item",y, ", output shape",img_out.shape)
        img_out = img_out.transpose(1,2,0)
        for i in range(0, 200*batch_size-1):  # there is returned 200 detections for each image in the batch
            detection = output[:,:,i,:]
            # each detection has shape 1,1,7 where last dimension represent:
            # image_id - ID of the image in the batch
            # label - predicted class ID
            # conf - confidence for the predicted class
            # (x_min, y_min) - coordinates of the top left bounding box corner
            #(x_max, y_max) - coordinates of the bottom right bounding box corner.
            if detection[0,0,2] > 0.5 and int(detection[0,0,0]) == y:  # ignore detections for image_id != y and confidence <0.5
                print("detection", i , detection)
                x_min = int(detection[0,0,3] * args['width'])
                y_min = int(detection[0,0,4] * args['height'])
                x_max = int(detection[0,0,5] * args['width'])
                y_max = int(detection[0,0,6] * args['height'])
                # box coordinates are proportional to the image size
                print("x_min", x_min)
                print("y_min", y_min)
                print("x_max", x_max)
                print("y_max", y_max)

                img_out = cv2.rectangle(cv2.UMat(img_out),(x_min,y_min),(x_max,y_max),(0,0,255),1)
                # draw each detected box on the input image

        output_path = os.path.join(args['output_dir'],model_name+"_"+str(iteration)+"_"+str(y)+'.jpg')
        print("saving result to", output_path)
        result_flag = cv2.imwrite(output_path,img_out)
        print("write success = ", result_flag)

    print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'
          .format(iteration, round(np.average(duration), 2), round(1000 * batch_size / np.average(duration), 2)
                                                                                  ))

print_statistics(processing_times, batch_size)

