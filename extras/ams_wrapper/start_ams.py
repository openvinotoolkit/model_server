#!/usr/bin/env python3
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
import json
import subprocess
import sys
import os


def parse_ovms_model_devices_config(config: str) -> dict:
    if not config:
        return {}
    try:
        return {
            model_name: device for model_name, device in [item.split('=') for item in config.split(';')]
        }
    except Exception as e:
        print('Invalid model devices config: {}'.format(config))
        raise ValueError from e


def modify_ovms_config_json(devices_config: dict,
                            ovms_config_path: str = '/opt/ams_models/ovms_config.json'):
    with open(ovms_config_path, mode='r') as ovms_config_file:
        ovms_config = json.load(ovms_config_file)
        for model_config in ovms_config.get('model_config_list'):
            model_config = model_config['config']
            if devices_config.get(model_config['name']):
                model_config['target_device'] = devices_config[model_config['name']]
    with open(ovms_config_path, mode='w') as ovms_config_file:
        json.dump(ovms_config, ovms_config_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script runs OpenVINO Model Server and AMS Service in the background. '
                    'OVMS will served models available under path /opt/models with configuration '
                    'defined in /opt/models/config.json file. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ams_port', type=int, default=5000,
                        help='Port for AMS Service to listen on')
    parser.add_argument('--ovms_port', type=int, default=9000,
                        help='Port for OVMS to listen on')
    parser.add_argument('--workers', type=int, default=20,
                        help='AMS service workers')
    parser.add_argument('--grpc_workers', type=int, default=10,
                        help='OVMS service workers')
    parser.add_argument('--ovms_model_devices', type=str,
                        help='Colon delimited list of model devices, '
                        'in following format: \'<model_1_name>=<device_name>;<model_2_name>=<device_name>;\'',
                        default=os.environ.get('OVMS_MODEL_DEVICES', ''))
    args = parser.parse_args()
    args.ovms_model_devices = parse_ovms_model_devices_config(
        args.ovms_model_devices)
    return args


def main():
    args = parse_args()
    if args.ovms_model_devices:
        modify_ovms_config_json(args.ovms_model_devices)
    ovms_process = subprocess.Popen(['/ie-serving-py/start_server.sh', 'ie_serving', 'config',
                                     '--config_path', '/opt/ams_models/ovms_config.json',
                                     '--grpc_workers', str(args.grpc_workers),
                                     '--port', str(args.ovms_port)])
    ams_process = subprocess.Popen(['/ie-serving-py/.venv/bin/python', '-m', 'src.wrapper', '--port',
                                    str(args.ams_port), '--workers', str(args.workers)],
                                   cwd='/ams_wrapper')
    retcodes = [ovms_process.wait(), ams_process.wait()]
    sys.exit(max(retcodes))


if __name__ == "__main__":
    main()
