#
# Copyright (c) 2020 Intel Corporation
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
import sys

from models.classifier_model import ClassifierModel
from models.detection_model import DetectionModel


def run_tests(model_type, ams_address, ams_port, ams_endpoint, ovms_address,
              ovms_port, ovms_model_name, image_path, model_json):
    if model_type == 'classifier':
        testing_model = ClassifierModel(ams_address, ams_port,
                                        ams_endpoint, ovms_address, ovms_port,
                                        ovms_model_name, image_path, model_json)
    elif model_type == 'detection':
        testing_model = DetectionModel(ams_address, ams_port,
                                       ams_endpoint, ovms_address, ovms_port,
                                       ovms_model_name, image_path, model_json)
    else:
        print("Invalid model type selected {}. Currently supported types: classifier/detection".format(model_type))
        sys.exit()
    testing_model.input_preprocess()
    testing_model.send_data()
    testing_model.output_postprocess()
    testing_model.print_results()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str,
                        help='Model type (classifier/detection)',
                        required=True)
    parser.add_argument('--ams_address', type=str,
                        help='AMS service listening address',
                        required=False, default='localhost')
    parser.add_argument('--ams_port', type=int,
                        help='AMS service listening port',
                        required=False, default=5000)
    parser.add_argument('--ams_endpoint', type=str,
                        help='AMS model endpoint',
                        required=True)
    parser.add_argument('--ovms_address', type=str,
                        help='OpenVINO Model Server listening address',
                        required=False, default='localhost')
    parser.add_argument('--ovms_port', type=int,
                        help='OpenVINO Model Server port',
                        required=False, default=9000)
    parser.add_argument('--ovms_model_name', type=str,
                        help='Name of OVMS model',
                        required=True)
    parser.add_argument('--image_path', type=str,
                        help='Path to testing image',
                        required=True)
    parser.add_argument('--model_json', type=str,
                        help='Path to model JSON file',
                        required=True)

    args = parser.parse_args()
    run_tests(args.model_type, args.ams_address, args.ams_port,
              args.ams_endpoint, args.ovms_address, args.ovms_port,
              args.ovms_model_name, args.image_path, args.model_json)


main()
