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
import re
import shutil
from datetime import datetime
from pathlib import Path

import logging
from tests.functional.utils.parametrization import get_tests_suffix, generate_test_object_name
from tests.functional.utils.process import Process

from tests.functional.config import converted_models_expire_time

logger = logging.getLogger(__name__)

def wget_file(url, dst):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading file via wget\n{url} => {dst}")
    proc = Process()
    proc.set_log_silence()
    cmd = f"wget {url} -O {dst}"
    proc.policy['log-check-output']['stderr'] = False
    proc.run_and_check(cmd)


def _get_file_size_from_url(url):
    cmd = f"wget -e robots=off --spider -r --server-response --no-parent {url}"
    # .group(1) - url
    # .group(2) - wget spider details
    # .group(3) - Content-Lenght:
    spider_entry_re = re.compile("--  (https?://[\S]+)\n(.+?)Content-Length: (\S+)", flags=re.MULTILINE | re.DOTALL)
    proc = Process()
    proc.set_log_silence()
    proc.policy['log-check-output']['stderr'] = False
    code, out, err = proc.run_and_check_return_all(cmd)

    all_entries_match = spider_entry_re.findall(err)
    # filter only files (entries specified Length: )
    model_files_match = list(filter(lambda x: x[2] != "unspecified" and x[2] != '0', all_entries_match))
    assert(len(model_files_match) == 1)
    return int(model_files_match[0][2])

def download_missing_file(url, output_path):
    size = _get_file_size_from_url(url)
    while not Path(output_path).exists() or Path(output_path).stat().st_size != size:
        wget_file(url, output_path)
    return output_path

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
                  input_shape,
                  framework="tf"):


    files = (os.path.join(output_dir, model_name) + '.bin',
             os.path.join(output_dir, model_name) + '.xml')

    # Check if file exists and is not expired
    if all(map(lambda x: os.path.exists(x) and \
                         datetime.now().timestamp() - os.path.getmtime(x) < converted_models_expire_time,
               files)):
        return files

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_shape_str = '[{}]'.format(','.join(str(i) for i in input_shape))
    logger.info("Converting {model} to IR with input shape {input_shape}...".format(model=model,
                                                                                    input_shape=input_shape_str))

    input_dir = os.path.dirname(model)

    image = 'openvino/ubuntu20_dev:2022.1.0'
    volumes = {input_dir:   {'bind': '/mnt/input_dir',  'mode': 'ro'},
               output_dir:  {'bind': '/mnt/output_dir', 'mode': 'rw'}}
    user_id = os.getuid()

    command = ' '.join([
        'mo',
        '--input_model /mnt/input_dir/' + os.path.basename(model),
        '--model_name ' + model_name,
        '--output_dir /mnt/output_dir/',
        '--input_shape ' + input_shape_str,
        '--framework ' + framework
    ])

    client.containers.run(image=image,
                          name='convert-model-{}-{}'.format(get_tests_suffix(), generate_test_object_name(short=True)),
                          volumes=volumes,
                          user=user_id,
                          command=command,
                          remove=True)
    return files
