#
# Copyright (c) 2019 Intel Corporation
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

import numpy as np
import classes
import datetime
import argparse
import json
import requests
from client_utils import print_statistics


def create_request(img, request_format):
    signature = "serving_default"
    if request_format == "row_name":
        instances = []
        for i in range(0, img.shape[0], 1):
            instances.append({args['input_name']: img[i].tolist()})
        data_obj = {"signature_name": signature, "instances": instances}
    elif request_format == "row_noname":
        data_obj = {"signature_name": signature, 'instances': img.tolist()}
    elif request_format == "column_name":
        data_obj = {"signature_name": signature,
                    'inputs': {args['input_name']: img.tolist()}}
    elif request_format == "column_noname":
        data_obj = {"signature_name": signature, 'inputs':  img.tolist()}
    else:
        print("invalid request format defined")
        exit(1)
    data_json = json.dumps(data_obj)
    return data_json


parser = argparse.ArgumentParser(description='Sends requests via TensorFlow Serving RESTfull API using images in numpy format. '
                                             'It displays performance statistics and optionally the model accuracy')
parser.add_argument('--images_numpy_path', required=True, help='numpy in shape [n,w,h,c] or [n,c,h,w]')
parser.add_argument('--labels_numpy_path', required=False, help='numpy in shape [n,1] - can be used to check model accuracy')
parser.add_argument('--rest_url', required=False, default='http://localhost',  help='Specify url to REST API service. Default: http://localhost')
parser.add_argument('--rest_port', required=False, default=8000, help='Specify port to REST API service. Default: 8000')
parser.add_argument('--input_name', required=False, default='input', help='Specify input tensor name. Default: input')
parser.add_argument('--output_name', required=False, default='resnet_v1_50/predictions/Reshape_1',
                    help='Specify output name. Default: resnet_v1_50/predictions/Reshape_1')
parser.add_argument('--transpose_input', choices=["False", "True"], default="True",
                    help='Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. Default: True',
                    dest="transpose_input")
parser.add_argument('--transpose_method', choices=["nchw2nhwc", "nhwc2nchw"], default="nhwc2nchw",
                    help="How the input transposition should be executed: nhwc2nchw or nhwc2nchw",
                    dest="transpose_method")
parser.add_argument('--iterations', default=0,
                    help='Number of requests iterations, as default use number of images in numpy memmap. Default: 0 (consume all frames)',
                    dest='iterations', type=int)
# If input numpy file has too few frames according to the value of iterations and the batch size, it will be
# duplicated to match requested number of frames
parser.add_argument('--batchsize', default=1,
                    help='Number of images in a single request. Default: 1',
                    dest='batchsize')
parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. Default: resnet',
                    dest='model_name')
parser.add_argument('--request_format', default='row_noname', help='Request format according to TF Serving API: row_noname,row_name,column_noname,column_name',
                    choices=["row_noname", "row_name", "column_noname", "column_name"], dest='request_format')
parser.add_argument('--model_version', help='Model version to be used. Default: LATEST',
                    type=int, dest='model_version')
parser.add_argument('--client_cert', required=False, default=None, help='Specify mTLS client certificate file. Default: None.')
parser.add_argument('--client_key', required=False, default=None, help='Specify mTLS client key file. Default: None.')
parser.add_argument('--ignore_server_verification', required=False, action='store_true', help='Skip TLS host verification. Do not use in production. Default: False.')
parser.add_argument('--server_cert', required=False, default=None, help='Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. Default: None, will use default system CA cert store.')

args = vars(parser.parse_args())

certs = None
verify_server = None
if args.get('client_cert') is not None or args.get('client_key') is not None:
  if args.get('client_cert') is not None and args.get('client_key') is not None and args.get('rest_url').startswith("https"):
    certs = (args.get('client_cert'), args.get('client_key'))
    if args.get('server_cert') is not None:
      verify_server = args.get('server_cert')
    if args.get('ignore_server_verification') is True:
      verify_server = False
  else:
    print("Error: in order to use mTLS, you need to provide both --client_cert and --client_key. In addition, your --rest_url flag has to begin with 'https://'.")
    exit(1)

processing_times = np.zeros((0), int)

# optional preprocessing depending on the model
imgs = np.load(args['images_numpy_path'], mmap_mode='r', allow_pickle=False)
imgs = imgs - np.min(imgs)  # Normalization 0-255
imgs = imgs / np.ptp(imgs) * 255  # Normalization 0-255
# imgs = imgs[:,:,:,::-1] # RGB to BGR
imgs = imgs.astype(np.uint8)
print('Image data range:', np.amin(imgs), ':', np.amax(imgs))
# optional preprocessing depending on the model

if args.get('labels_numpy_path') is not None:
    lbs = np.load(args['labels_numpy_path'], mmap_mode='r', allow_pickle=False)
    matched_count = 0
    total_executed = 0
batch_size = int(args.get('batchsize'))


while batch_size >= imgs.shape[0]:
    imgs = np.append(imgs, imgs, axis=0)
    if args.get('labels_numpy_path') is not None:
        lbs = np.append(lbs, lbs, axis=0)

iterations = int((imgs.shape[0]//batch_size) if not (args.get('iterations') or args.get('iterations') != 0) else args.get('iterations'))

print('Start processing:')
print('\tModel name: {}'.format(args.get('model_name')))
print('\tIterations: {}'.format(iterations))
print('\tImages numpy path: {}'.format(args.get('images_numpy_path')))

if args.get('transpose_input') == "True":
    if args.get('transpose_method') == "nhwc2nchw":
        imgs = imgs.transpose((0, 3, 1, 2))
    if args.get('transpose_method') == "nchw2nhwc":
        imgs = imgs.transpose((0, 2, 3, 1))
print('\tImages in shape: {}\n'.format(imgs.shape))

iteration = 0

session = requests.Session()
while iteration <= iterations:
    for x in range(0, imgs.shape[0] - batch_size + 1, batch_size):
        iteration += 1
        if iteration > iterations:
            break

        img = imgs[x:(x + batch_size)]
        if args.get('labels_numpy_path') is not None:
            lb = lbs[x:(x + batch_size)]
        data_json = create_request(img, args.get('request_format'))
        version = ""
        if args.get('model_version') is not None:
            version = "/versions/{}".format(args.get('model_version'))
        start_time = datetime.datetime.now()
        result = session.post("{}:{}/v1/models/{}{}:predict".format(args['rest_url'], args['rest_port'], args['model_name'], version), data=data_json, cert=certs, verify=verify_server)
        end_time = datetime.datetime.now()
        try:
            result_dict = json.loads(result.text)
        except ValueError:
            print("The server response is not json format: {}",format(result.text))
            exit(1)
        if "error" in result_dict:
            print('Server returned error: {}'.format(result_dict))
            exit(1)

        if "outputs" in result_dict:  # is column format
            keyname = "outputs"
            if type(result_dict[keyname]) is dict:
                if args['output_name'] not in result_dict[keyname]:
                    print("Invalid output name", args['output_name'])
                    print("Available outputs:")
                    for Y in result_dict[keyname]:
                        print(Y)
                    exit(1)
                output = result_dict[keyname][args['output_name']]
            else:
                output = result_dict[keyname]
        elif "predictions" in result_dict:  # is row format
            keyname = "predictions"
            if type(result_dict[keyname][0]) is dict:  # are multiple outputs
                output = []
                for row in result_dict[keyname]:  # iterate over all results in the batch
                    output.append(row[args['output_name']])
            else:
                output = result_dict[keyname]
        else:
            print("Missing required response in {}".format(result_dict))
            exit(1)

        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times, np.array([int(duration)]))
        # print(output)
        nu = np.array(output)  # numpy array with inference results
        print("output shape: {}".format(nu.shape))

        # for object classification models show imagenet class
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration, round(np.average(duration), 2),
                                                                                  round(1000 * batch_size / np.average(duration), 2)))
        # Comment out this section for non imagenet datasets
        print("imagenet top results in a single batch:")
        for i in range(nu.shape[0]):
            single_result = nu[[i], ...]
            ma = np.argmax(single_result)
            mark_message = ""
            if args.get('labels_numpy_path') is not None:
                total_executed += 1
                if ma == lb[i]:
                    matched_count += 1
                    mark_message = "; Correct match."
                else:
                    mark_message = "; Incorrect match. Should be {} {}".format(lb[i], classes.imagenet_classes[lb[i]])
            print("\t", i, classes.imagenet_classes[ma], ma, mark_message)
        # Comment out this section for non imagenet datasets

print_statistics(processing_times, batch_size)

if args.get('labels_numpy_path') is not None:
    print('Classification accuracy: {:.2f}'.format(100*matched_count/total_executed))
