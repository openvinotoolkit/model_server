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

from ovmsclient import make_grpc_metadata_request
import cv2
import numpy as np


def read_paths(images_dir):
    with open(images_dir) as f:
        return f.readlines()


def load_image(path):
    with open(path, 'rb') as f:
        return f.read()


# get single input and output name from model metadata
def get_single_metadata(client, model_name, model_version):
    metadata_request = make_grpc_metadata_request(model_name, model_version)
    metadata = client.get_model_metadata(metadata_request)
    metadata_dict = metadata.to_dict()
    version = next(iter(metadata_dict))
    input_name = next(iter(metadata_dict[version]['inputs']))  # by default resnet has one input and one output
    output_name = next(iter(metadata_dict[version]['outputs']))
    return input_name, output_name


# read .txt file with listed images into ndarray in model shape
def read_imgs_as_ndarray(images_dir, height, width):
    with open(images_dir) as f:
        paths = f.readlines()
    imgs = np.zeros((0, 3, height, width))
    for path in paths:
        path = path.strip()
        img = getJpeg(path, height, width)
        imgs = np.append(imgs, img, axis=0)
    return imgs.astype('float32')


def crop_resize(img,cropx,cropy):
    y, x, c = img.shape
    if y < cropy:
        img = cv2.resize(img, (x, cropy))
        y = cropy
    if x < cropx:
        img = cv2.resize(img, (cropx, y))
        x = cropx
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def getJpeg(path, height, width, rgb_image=0):
    with open(path, mode='rb') as file:
        content = file.read()

    img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
    # retrived array has BGR format and 0-255 normalization
    # format of data is HWC
    # add image preprocessing if needed by the model
    img = crop_resize(img, height, width)
    img = img.astype('float32')
    #convert to RGB instead of BGR if required by model
    if rgb_image:
        img = img[:, :, [2, 1, 0]]
    # switch from HWC to CHW and reshape to 1,3,size,size for model blob input requirements
    img = img.transpose(2, 0, 1).reshape(1, 3, height, width)
    return img
