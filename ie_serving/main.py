#
# Copyright (c) 2018-2019 Intel Corporation
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
import sys
import threading

from ie_serving.models.model_builder import ModelBuilder
from ie_serving.server.start import serve as start_server
from ie_serving.logger import get_logger, LOGGER_LVL
from ie_serving.server.start import start_web_rest_server
from jsonschema.exceptions import ValidationError
import os

logger = get_logger(__name__)


def open_config(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Error occurred while opening config file {}".format(e))
        sys.exit()
    return data


def check_config_structure(configs):
    required_keys = ('base_path', 'name')
    if 'model_config_list' in configs:
        try:
            for config in configs['model_config_list']:
                if 'config' in config:
                    if not all(k in config['config'] for k in required_keys):
                        logger.error("Config objects in the config file must"
                                     " contain 'base_path' and 'name' strings")
                        sys.exit()
                else:
                    logger.error("'model_config_list' array in the config"
                                 "must contain 'config' object")
                    sys.exit()
        except Exception as e:
            logger.error("Error occurred while parsing config file: "
                         "{}".format(e))
            sys.exit()
    else:
        logger.error("Config file must contain 'model_config_list' array")
        sys.exit()


def parse_config(args):
    configs = open_config(path=args.config_path)
    check_config_structure(configs=configs)
    models = {}
    for config in configs['model_config_list']:
        try:
            batch_size = config['config'].get('batch_size', None)
            reshapable = config['config'].get('reshapable', False)

            model_ver_policy = config['config'].get(
                'model_version_policy', None)
            model = ModelBuilder.build(model_name=config['config']['name'],
                                       model_directory=config['config'][
                                           'base_path'],
                                       batch_size=batch_size,
                                       reshapable=reshapable,
                                       model_version_policy=model_ver_policy)
            if model is not None:
                models[config['config']['name']] = model
                if reshapable and batch_size is not None:
                    logger.warning(
                        "Both reshapable and batch_size parameters are set "
                        "for model: {}. Assuming that model is reshapable - "
                        "batch_size will be ignored".format(model.model_name))
        except ValidationError as e_val:
            logger.warning("Model version policy for model {} is invalid. "
                           "Exception: {}".format(config['config']['name'],
                                                  e_val))
        except Exception as e:
            logger.warning("Unexpected error occurred in {} model. "
                           "Exception: {}".format(config['config']['name'],
                                                  e))
    if not models:
        logger.info("Could not access any of provided models. Server will "
                    "exit now.")
        sys.exit()
    if args.rest_port > 0:
        process_thread = threading.Thread(target=start_web_rest_server,
                                          args=[models, args.rest_port])
        process_thread.setDaemon(True)
        process_thread.start()
    start_server(models=models, max_workers=1, port=args.port)


def parse_one_model(args):
    try:
        model_version_policy = json.loads(args.model_version_policy)
        model = ModelBuilder.build(model_name=args.model_name,
                                   model_directory=args.model_path,
                                   batch_size=args.batch_size,
                                   reshapable=args.reshapable,
                                   model_version_policy=model_version_policy)
    except ValidationError as e_val:
        logger.error("Model version policy is invalid. "
                     "Exception: {}".format(e_val))
        sys.exit()
    except json.decoder.JSONDecodeError as e_json:
        logger.error("model_version_policy field must be in json format. "
                     "Exception: {}".format(e_json))
        sys.exit()
    except Exception as e:
        logger.error("Unexpected error occurred. "
                     "Exception: {}".format(e))
        sys.exit()
    models = {}
    if model is not None:
        models[args.model_name] = model
        if args.reshapable and args.batch_size is not None:
            logger.warning(
                "Both reshapable and batch_size parameters are set "
                "for model: {}. Assuming that model is reshapable - "
                "batch_size will be ignored".format(model.model_name))
    else:
        logger.info("Could not access provided model. Server will exit now.")
        sys.exit()

    if args.rest_port > 0:
        process_thread = threading.Thread(target=start_web_rest_server,
                                          args=[models, args.rest_port])
        process_thread.setDaemon(True)
        process_thread.start()
    start_server(models=models, max_workers=1, port=args.port)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('config',
                                     help='Allows you to share multiple '
                                          'models using a configuration file')
    parser_a.add_argument('--config_path', type=str,
                          help='absolute path to json configuration file',
                          required=True)
    parser_a.add_argument('--port', type=int, help='gRPC server port',
                          required=False, default=9000)
    parser_a.add_argument('--rest_port', type=int,
                          help='REST server port, the REST server will not be'
                               ' started if rest_port is blank or set to 0',
                          required=False, default=0)
    parser_a.set_defaults(func=parse_config)

    parser_b = subparsers.add_parser('model',
                                     help='Allows you to share one type of '
                                          'model')
    parser_b.add_argument('--model_name', type=str, help='name of the model',
                          required=True)
    parser_b.add_argument('--model_path', type=str,
                          help='absolute path to model,as in tf serving',
                          required=True)
    parser_b.add_argument('--batch_size', type=str,
                          help='sets models batchsize, int value or auto. '
                               'This parameter will be ignored if '
                               'reshapable flag is set.', required=False)
    parser_b.add_argument('--reshapable',
                          help='Enables reshaping for the model '
                               '(model must support reshaping). If set, '
                               'batch_size parameter is ignored.',
                          required=False, action='store_true')
    parser_b.add_argument('--port', type=int, help='gRPC server port',
                          required=False, default=9000)
    parser_b.add_argument('--rest_port', type=int,
                          help='REST server port, the REST server will not be'
                               ' started if rest_port is blank or set to 0',
                          required=False, default=0)
    parser_b.add_argument('--model_version_policy', type=str,
                          help='model version policy',
                          required=False,
                          default='{"latest": { "num_versions":1 }}')
    parser_b.set_defaults(func=parse_one_model)
    args = parser.parse_args()
    logger.info("Log level set: {}".format(LOGGER_LVL))
    logger.debug("ie_serving_py arguments: {}".format(args))
    logger.debug("configured environment variables: {}".format(os.environ))
    args.func(args)


if __name__ == '__main__':
    main()
