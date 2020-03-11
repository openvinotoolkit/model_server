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

import os
import shutil
import time

from pathlib import Path


def wait_endpoint_setup(container):
    start_time = time.time()
    tick = start_time
    running = False
    logs = ""
    while tick - start_time < 300:
        tick = time.time()
        try:
            logs = str(container.logs())
            if "server listens on port" in logs:
                running = True
                break
        except Exception as e:
            time.sleep(1)
    print("Logs from container: ", logs)
    return running


def copy_model(model, version, destination_path):
    dir_to_cpy = destination_path + str(version)
    if not os.path.exists(dir_to_cpy):
        os.makedirs(dir_to_cpy)
        shutil.copy(model[0], dir_to_cpy + '/model.bin')
        shutil.copy(model[1], dir_to_cpy + '/model.xml')
    return dir_to_cpy


def convert_model(client,
                  model,
                  output_dir,
                  model_name,
                  input_shape):

    files = (os.path.join(output_dir, model_name) + '.bin',
             os.path.join(output_dir, model_name) + '.xml')

    if os.path.exists(files[0]) and os.path.exists(files[1]):
        return files

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_shape_str = '[{}]'.format(','.join(str(i) for i in input_shape))
    print("Converting {} to IR with input shape {}...".format(model, input_shape_str))

    input_dir = os.path.dirname(model)

    image = 'openvino/ubuntu18_dev:latest'
    volumes = {input_dir:   {'bind': '/mnt/input_dir',  'mode': 'ro'},
               output_dir:  {'bind': '/mnt/output_dir', 'mode': 'rw'}}

    command = ' '.join([
        'python3 deployment_tools/model_optimizer/mo.py',
        '--input_model /mnt/input_dir/' + os.path.basename(model),
        '--model_name ' + model_name,
        '--output_dir /mnt/output_dir/',
        '--input_shape ' + input_shape_str
    ])

    client.containers.run(image=image,
                          volumes=volumes,
                          command=command)

    return files
