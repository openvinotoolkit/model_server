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

from ovmsclient import make_grpc_client
import cv2
import numpy as np
import argparse


colors = [
    (128,  64, 128), # road
    (244,  35, 232), # sidewalk
    ( 70,  70,  70), # building
    (102, 102, 156), # wall
    (190, 153, 153), # fence
    (153, 153, 153), # pole
    (250, 170,  30), # traffic light
    (220, 220,   0), # traffic sign
    (107, 142,  35), # vegetation
    (152, 251, 152), # terrain
    ( 70, 130, 180), # sky
    (220,  20,  60), # person
    (255,   0,   0), # rider
    (  0,   0, 142), # car
    (  0,   0,  70), # truck
    (  0,  60, 100), # bus
    (  0,  80, 100), # train
    (  0,   0, 230), # motorcycle
    (119,  11,  32), # bicycle
    (255, 255, 255) # background
]

def load_img(path):
    img = cv2.imread(path)
    img_f = img.astype(np.float32)
    img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
    mean = [127.5,127.5,127.5]
    scale = [127.5,127.5,127.5]
    img_f = (img_f - np.array(mean, dtype=np.float32))/np.array(scale, dtype=np.float32)
    img_f = img_f.transpose(2,0,1).reshape(1,3,img_f.shape[0], img_f.shape[1])
    return img, {"x": img_f}

def get_mask(img, segmentation_output):
    mask = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mask[i, j] = colors[segmentation_output[0, i, j]]
    return mask

def build_parser():
    parser = argparse.ArgumentParser(description='Client for OCR pipeline')
    parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port', required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--image_input_path', required=True, help='Image input path')
    parser.add_argument('--image_output_path', required=True, help='Path to save segmented image')
    return parser

if __name__ == "__main__":
    args = vars(build_parser().parse_args())

    img_path = args['image_input_path']
    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
    img, input = load_img(img_path)

    client = make_grpc_client(address)
    segmentation_output = client.predict(input, "ocrnet")

    mask = get_mask(img, segmentation_output)
    img = cv2.addWeighted(img, .5, mask, .5, 0)
            
                
    cv2.imwrite(args['image_output_path'], img)

    