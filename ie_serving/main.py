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
import shutil
import sys
from urllib.parse import urlparse

from ie_serving.server.start import serve as start_server
from ie_serving.models.model import Model
from ie_serving.logger import get_logger, LOGGER_LVL
import os
import boto3
from google.cloud import storage

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


def s3_download_dir(client, bucket_name, source_directory, destination_directory):
    paginator = client.get_paginator('list_objects')
    for listing_result in paginator.paginate(Bucket=bucket_name, Delimiter='/',
                                             Prefix=source_directory):
        if listing_result.get('CommonPrefixes') is not None:
            for subdirectory in listing_result.get('CommonPrefixes'):
                s3_download_dir(client, bucket_name, subdirectory.get('Prefix'),
                                destination_directory)
        if listing_result.get('Contents') is not None:
            for file in listing_result.get('Contents'):
                download_file_path = os.path.join(destination_directory, file.get('Key'))
                download_directory_path = os.path.dirname(download_file_path)
                if not os.path.exists(download_directory_path):
                    os.makedirs(download_directory_path)
                client.download_file(bucket_name, file.get('Key'),
                                     download_file_path)


def s3_download_model(parsed_url):
    AWS_ACCESS_KEY = os.getenvb('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenvb('AWS_SECRET_KEY')
    bucket_name = parsed_url.netlock
    model_directory = parsed_url.path
    s3_client = boto3.client('s3',
                             endpoint_url='https://storage.googleapis.com',
                             aws_access_key_id=AWS_ACCESS_KEY,
                             aws_secret_access_key=AWS_SECRET_KEY)
    s3_download_dir(s3_client, bucket_name, model_directory, 'tmp')


def gs_download_model(parsed_url):
    bucket_name = parsed_url.netlock
    model_directory = parsed_url.path
    gs_client = storage.Client()
    bucket = gs_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_directory)
    for blob in blobs:
        if blob.name[-1] != os.sep:
            download_directory_path = os.path.join('tmp', os.path.dirname(blob.name))
            if not os.path.exists(download_directory_path):
                os.makedirs(download_directory_path)
            blob.download_to_filename(os.path.join('tmp', blob.name))


def load_model(config: dict):
    parsed_url = urlparse(config['base_path'])
    if parsed_url.scheme is '':
        model = Model.build(model_name=config['name'],
                            model_directory=config['base_path'])
    elif parsed_url.scheme is 's3':
        os.mkdir('tmp')
        s3_download_model(parsed_url)
        model = Model.build(model_name=config['name'],
                            model_directory='tmp')
        shutil.rmtree('tmp')
    elif parsed_url.scheme is 'gs':
        os.mkdir('tmp')
        gs_download_model(parsed_url)
        model = Model.build(model_name=config['name'],
                            model_directory='tmp')
        shutil.rmtree('tmp')
    else:
        print('Unavailable url scheme')
        return None
    return model


def parse_config(args):
    configs = open_config(path=args.config_path)
    check_config_structure(configs=configs)
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
    logger.info("Log level set: {}".format(LOGGER_LVL))
    logger.debug("ie_serving_py arguments: {}".format(args))
    logger.debug("configured environment variables: {}".format(os.environ))
    args.func(args)


if __name__ == '__main__':
    main()
