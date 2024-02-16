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
import sys
sys.path.append("../../../../demos/common/python")

import numpy as np
import classes
import datetime
import argparse
from client_utils import print_statistics, prepare_certs
import tritonclient.http as httpclient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe REST API using images in numpy format. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--images_numpy_path', required=True, help='numpy in shape [n,w,h,c] or [n,c,h,w]')
    parser.add_argument('--labels_numpy_path', required=False, help='numpy in shape [n,1] - can be used to check model accuracy')
    parser.add_argument('--http_address',required=False, default='localhost',  help='Specify url to http service. default:localhost')
    parser.add_argument('--http_port',required=False, default=8000, help='Specify port to http service. default: 8000')
    parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
    parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1',
                        help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
    parser.add_argument('--transpose_input', choices=["False", "True"], default="True",
                        help='Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. default: True',
                        dest="transpose_input")
    parser.add_argument('--transpose_method', choices=["nchw2nhwc","nhwc2nchw"], default="nhwc2nchw",
                        help="How the input transposition should be executed: nhwc2nchw or nchw2nhwc",
                        dest="transpose_method")
    parser.add_argument('--iterations', default=0,
                        help='Number of requests iterations, as default use number of images in numpy memmap. default: 0 (consume all frames)',
                        dest='iterations', type=int)
    # If input numpy file has too few frames according to the value of iterations and the batch size, it will be
    # duplicated to match requested number of frames
    parser.add_argument('--batchsize', default=1,
                        help='Number of images in a single request. default: 1',
                        dest='batchsize')
    parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                        dest='model_name')
    parser.add_argument('--pipeline_name', default='', help='Define pipeline name, must be same as is in service',
                        dest='pipeline_name')
    parser.add_argument('--dag-batch-size-auto', default=False, action='store_true', help='Add demultiplexer dimension at front', dest='dag-batch-size-auto')
    parser.add_argument('--binary_data', default=False, action='store_true', help='Send input data in binary format', dest='binary_data')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
    parser.add_argument('--server_cert', required=False, help='Path to server certificate', default=None)
    parser.add_argument('--client_cert', required=False, help='Path to client certificate', default=None)
    parser.add_argument('--client_key', required=False, help='Path to client key', default=None)

    args = vars(parser.parse_args())

    address = "{}:{}".format(args['http_address'],args['http_port'])

    if args['tls']:
        ssl_options = {
            'keyfile':args['client_key'],
            'cert_file':args['client_cert'],
            'ca_certs':args['server_cert']
        }
    else:
        ssl_options = None


    triton_client = httpclient.InferenceServerClient(
                url=address,
                ssl=args['tls'],
                ssl_options=ssl_options,
                verbose=False)

    processing_times = np.zeros((0),int)

    # optional preprocessing depending on the model
    imgs = np.load(args['images_numpy_path'], mmap_mode='r', allow_pickle=False)
    imgs = imgs - np.min(imgs)  # Normalization 0-255
    imgs = imgs / np.ptp(imgs) * 255  # Normalization 0-255
    #imgs = imgs[:,:,:,::-1] # RGB to BGR
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
    print('\tModel name: {}'.format(args.get('pipeline_name') if bool(args.get('pipeline_name')) else args.get('model_name')))
    print('\tIterations: {}'.format(iterations))
    print('\tImages numpy path: {}'.format(args.get('images_numpy_path')))
    if args.get('transpose_input') == "True":
        if args.get('transpose_method') == "nhwc2nchw":
            imgs = imgs.transpose((0,3,1,2))
        if args.get('transpose_method') == "nchw2nhwc":
            imgs = imgs.transpose((0,2,3,1))
    print('\tNumpy file shape: {}\n'.format(imgs.shape))

    iteration = 0
    is_pipeline_request = bool(args.get('pipeline_name'))

    while iteration <= iterations:
        for x in range(0, imgs.shape[0] - batch_size + 1, batch_size):
            iteration += 1
            if iteration > iterations: break
            img = imgs[x:(x + batch_size)]
            if args.get('labels_numpy_path') is not None:
                lb = lbs[x:(x + batch_size)]
            inputs = []
            if args.get('dag-batch-size-auto'):
                newShape = img.shape[0:1] + (1,) + img.shape[1:]
                inputs.append(httpclient.InferInput(args['input_name'], newShape, "FP32"))
            else:
                inputs.append(httpclient.InferInput(args['input_name'], img.shape, "FP32"))
            outputs = []
            inputs[0].set_data_from_numpy(img, binary_data=args.get('binary_data'))
            start_time = datetime.datetime.now()
            results = triton_client.infer(
                model_name=args.get('pipeline_name') if is_pipeline_request else args.get('model_name'),
                inputs=inputs)
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            processing_times = np.append(processing_times,np.array([int(duration)]))
            output = results.as_numpy(args['output_name'])
            nu = np.array(output)
            # for object classification models show imagenet class
            print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration,round(np.average(duration), 2),
                                                                                      round(1000 * batch_size / np.average(duration), 2)
                                                                                      ))
            # Comment out this section for non imagenet datasets
            print(results)
            print("imagenet top results in a single batch:")
            for i in range(nu.shape[0]):
                if is_pipeline_request:
                    # shape (1,)
                    print("response shape", output.shape)
                    ma = nu[0] - 1 # indexes needs to be shifted left due to 1x1001 shape
                else:
                    # shape (1,1000)
                    single_result = nu[[i],...]
                    offset = 0
                    if nu.shape[1] == 1001:
                        offset = 1
                    ma = np.argmax(single_result) - offset
                mark_message = ""
                if args.get('labels_numpy_path') is not None:
                    total_executed += 1
                    if ma == lb[i]:
                        matched_count += 1
                        mark_message = "; Correct match."
                    else:
                        mark_message = "; Incorrect match. Should be {} {}".format(lb[i], classes.imagenet_classes[lb[i]] )
                print("\t", i, classes.imagenet_classes[ma], ma, mark_message)
            # Comment out this section for non imagenet datasets

    print_statistics(processing_times, batch_size)

    if args.get('labels_numpy_path') is not None:
        print('Classification accuracy: {:.2f}'.format(100*matched_count/total_executed))

