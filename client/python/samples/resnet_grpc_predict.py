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

import argparse
import numpy as np
import classes
from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client
from ovmsclient.tfs_compat.grpc.requests import make_predict_request

parser = argparse.ArgumentParser(description='Make prediction using images in numpy format')
parser.add_argument('--images_numpy_path', required=False, default='./utils/imgs.npy',
                    help='path to image')
parser.add_argument('--labels_numpy_path', required=False,
                    help='numpy in shape [n,1] - can be used to check model accuracy')
parser.add_argument('--grpc_address', required=False, default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='resnet', help='Model name to query. default: resnet',
                    dest='model_name')
parser.add_argument('--model_version', default=0, type=int,
                    help='Model version to query. Lists all versions if omitted',
                    dest='model_version')
parser.add_argument('--input_name', required=False, default='0',
                    help='Specify input tensor name. default: 0')
parser.add_argument('--output_name', required=False, default='1463',
                    help='Specify output name. default: 1463')
parser.add_argument('--batchsize', default=1, type=int,
                    help='Number of images in a single request. default: 1',
                    dest='batchsize')
args = vars(parser.parse_args())

# configuration
images_numpy_path = args.get('images_numpy_path')
labels_numpy_path = args.get('labels_numpy_path')
address = args.get('grpc_address')
port = args.get('grpc_port')
model_name = args.get('model_name')
model_version = args.get('model_version')
input_name = args.get('input_name')
output_name = args.get('output_name')
batch_size = args.get('batchsize')


# images pre-processing
# returns procssed images and potentially processed labels
def pre_processing():
    imgs = np.load(args['images_numpy_path'], mmap_mode='r', allow_pickle=False)
    imgs = imgs - np.min(imgs)  # Normalization 0-255
    imgs = imgs / np.ptp(imgs) * 255  # Normalization 0-255

    lbs = None
    if labels_numpy_path is not None:
        lbs = np.load(labels_numpy_path, mmap_mode='r', allow_pickle=False)

    while batch_size >= imgs.shape[0]:
        imgs = np.append(imgs, imgs, axis=0)
        if args.get('labels_numpy_path') is not None:
            lbs = np.append(lbs, lbs, axis=0)
    return imgs, lbs


def post_processing(output):
    for i in range(output.shape[0]):
        single_result = output[[i], ...]
        offset = 0
        if output.shape[1] == 1001:
            offset = 1
        ma = np.argmax(single_result) - offset
        mark_message = ""
        if labels_numpy_path is not None:
            if int(lb[i]) == ma:
                mark_message = "; Correct match."
            else:
                mark_message = "; Incorrect match. Should be {} {}".format(lb[i], classes.imagenet_classes[int(lb[i])])
        print(classes.imagenet_classes[ma], ma, mark_message)


# creating grpc client
config = {
    "address": address,
    "port": port
}
client = make_grpc_client(config)

# images pre-processing
imgs, lbs = pre_processing()

for x in range(0, imgs.shape[0] - batch_size + 1, batch_size):
    # images and labels batch size refactoring
    img = imgs[x:(x + batch_size)]
    if labels_numpy_path is not None:
        lb = lbs[x:(x + batch_size)]

    # preparing predict request
    inputs = {
        input_name: img
    }
    request = make_predict_request(inputs, model_name, model_version)
    response = client.predict(request)
    response_dict = response.to_dict()

    # output post-processing
    if output_name not in response_dict.keys():
        print(f"Invalid output name - {output_name}")
    output = response_dict[output_name]
    post_processing(output)
