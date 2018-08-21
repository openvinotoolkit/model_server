#
# Copyright (c) 2018 Intel Corporation
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
import json
from ie_serving.server.start import serve as start_server
from ie_serving.models.model import Model


def open_config(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def parse_config(args):
    configs = open_config(path=args.config_path)
    models = {}
    for config in configs['model_config_list']:
        model = Model.build(model_name=config['config']['name'],
                            model_directory=config['config']['base_path'])
        models[config['config']['name']] = model
    start_server(models=models, max_workers=1, port=args.port)


def parse_one_model(args):
    model = Model.build(model_name=args.model_name,
                        model_directory=args.model_path)
    start_server(models={args.model_name: model},
                 max_workers=1, port=args.port)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('config',
                                     help='Allows you to share multiple '
                                          'models using a configuration file')
    parser_a.add_argument('--config_path', type=str,
                          help='absolute path to json configuration file',
                          required=True)
    parser_a.add_argument('--port', type=int, help='server port',
                          required=False, default=9000)
    parser_a.set_defaults(func=parse_config)

    parser_b = subparsers.add_parser('model',
                                     help='Allows you to share one type of '
                                          'model')
    parser_b.add_argument('--model_name', type=str, help='name of the model',
                          required=True)
    parser_b.add_argument('--model_path', type=str,
                          help='absolute path to model,as in tf serving',
                          required=True)
    parser_b.add_argument('--port', type=int, help='server port',
                          required=False, default=9000)
    parser_b.set_defaults(func=parse_one_model)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
