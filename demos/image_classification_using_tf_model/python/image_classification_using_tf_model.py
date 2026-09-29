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

import tritonclient.grpc as grpcclient
import cv2
import numpy as np
import argparse
import sys
sys.path.append("../../common/python")
import classes



def load_img(path):
    img = cv2.imread(path)
    img_f = img.astype(np.float32)
    img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
    img_f = cv2.resize(img_f, (299,299))
    mean = [127.5,127.5,127.5]
    scale = [127.5,127.5,127.5]
    img_f = (img_f - np.array(mean, dtype=np.float32))/np.array(scale, dtype=np.float32)
    img_f = img_f.reshape(1, img_f.shape[0], img_f.shape[1], 3)
    return img_f

def build_parser():
    parser = argparse.ArgumentParser(description='Client for OCR pipeline')
    parser.add_argument('--grpc_address', required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port', required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--image_input_path', required=True, help='Image input path')
    parser.add_argument('--input_name', required=False, default='', help='Input tensor name. default: auto-detected from model metadata')
    parser.add_argument('--output_name', required=False, default='', help='Output tensor name. default: auto-detected from model metadata')
    return parser

if __name__ == "__main__":
    args = vars(build_parser().parse_args())

    img_path = args['image_input_path']
    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])

    client = grpcclient.InferenceServerClient(url=address)

    input_name = args['input_name']
    output_name = args['output_name']
    if not input_name or not output_name:
        metadata = client.get_model_metadata("resnet")
        if not input_name:
            input_name = metadata.inputs[0].name
        if not output_name:
            output_name = metadata.outputs[0].name

    img = load_img(img_path)
    infer_input = grpcclient.InferInput(input_name, img.shape, "FP32")
    infer_input.set_data_from_numpy(img)
    result = client.infer("resnet", [infer_input])
    classification_output = result.as_numpy(output_name)

    print("Image classified as " + classes.imagenet_classes[classification_output.argmax() - 1])
