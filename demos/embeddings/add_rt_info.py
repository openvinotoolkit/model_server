#
# Copyright (c) 2024 Intel Corporation
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

import argparse, os, tempfile, json, shutil
import openvino as ov
from openvino_tokenizers import convert_tokenizer, connect_models
parser = argparse.ArgumentParser(description='Add run time information to the model based on HF configuration')
parser.add_argument('--model_path', required=True, help='Path to OpenVINO XML file', dest='model_path')
parser.add_argument('--config_path', required=True, help='path to config json file with runtime into', dest='config_path')
args = vars(parser.parse_args())
model = ov.Core().read_model(model=args['model_path'])
with open(args['config_path'], 'r') as config_file:
    config_data = json.load(config_file)
    for key, value in config_data.items():
        try:
            model.set_rt_info(value, ['model_info', key])
        except Exception as e:
            model.set_rt_info(str(value), ['model_info', key])
with tempfile.TemporaryDirectory() as tmpdirname:
    ov.save_model(model, os.path.join(tmpdirname, os.path.splitext(os.path.basename(args['model_path']))[0]+'.xml'))
    shutil.copy(os.path.join(tmpdirname, os.path.splitext(os.path.basename(args['model_path']))[0]+'.xml'), os.path.dirname(args['model_path']))
    shutil.copy(os.path.join(tmpdirname, os.path.splitext(os.path.basename(args['model_path']))[0]+'.bin'), os.path.dirname(args['model_path']))
