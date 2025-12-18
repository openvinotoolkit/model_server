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

import os
import sys
import re
import subprocess

WIN_OV_VERSION_REGEX = re.compile(r'[0-9]{4}.[0-9].[0-9]+')
WIN_OV_ZIP_PACKAGE_DIR = "openvino_genai_windows_"
VERSION_FILE = "src\\version.hpp"
OVMS_PROJECT_VERSION="2026.0.0"

def help():
    print("Usage:\n\
          Two arguments required: BAZEL_BUILD_FLAGS PATH_TO_OPENVINO_INSTALL\n\
          for example: \n\
          python windows_set_ovms_version.py \"--config=windows\" c:\\opt\\ \n\
          ")

def escape_string(input):
    output = input.replace(r"\n",r"\\n").replace(r"\r",r"\\r")
    return output

def replace_in_file(file_path, old_string, new_string):
    new_string = escape_string(new_string)
    with open(file_path, 'r+') as file:
        contents = file.read()
        contents = contents.replace(old_string, new_string)
        file.seek(0)
        file.write(contents)
        file.truncate()

def get_openvino_name_bin(openvino_dir):
    openvino_name = "Unknown"
    # read file openvino_dir\openvino\runtime\version.txt
    version_file_path = os.path.join(openvino_dir, "openvino", "runtime", "version.txt")
    if os.path.exists(version_file_path):
        with open(version_file_path, 'r') as version_file:
            for line in version_file:
                match = WIN_OV_VERSION_REGEX.search(line)
                if match:
                    openvino_name = match.group(0)
                    break
    return openvino_name

def get_openvino_name_src(openvino_dir):
    src_dir = os.path.join(openvino_dir, "openvino_src")
    # get version using git
    command = "git -C {} rev-parse --short HEAD".format(src_dir)
    output = subprocess.check_output(command, shell=True, text=True)
    return output.rstrip()        
def get_ovms_sha():
    command = "git rev-parse --short HEAD"
    output = subprocess.check_output(command, shell=True, text=True)
    return output.rstrip()

def check_get_product_version():
    version = os.environ.get('PRODUCT_VERSION', OVMS_PROJECT_VERSION)
    return version

def main():
    if len(sys.argv) < 2:
        print('Provide bazel build flags!')
        help()
        exit(-1)
    else:
        bazel_bld_flags = sys.argv[1]
        print('Provided bazel build flags: ' + bazel_bld_flags)

    if len(sys.argv) < 3:
        print('Provide openvino directory!')
        help()
        exit(-1)
    else:
        openvino_dir = sys.argv[2]
        print('Provided openvino directory: ' + openvino_dir)

    if os.environ.get('OV_USE_BINARY', '1') == '1':
        print('Getting openvino version from binary package')
        openvino_name = get_openvino_name_bin(openvino_dir)
    else:
        openvino_name = get_openvino_name_src(openvino_dir)
        print('Using openvino source build, setting version to ' + openvino_name)
    version_file_path = os.path.join(os.getcwd(), VERSION_FILE)
    replace_in_file(version_file_path, "REPLACE_PROJECT_VERSION", check_get_product_version() + "." + get_ovms_sha())
    replace_in_file(version_file_path, "REPLACE_OPENVINO_NAME", openvino_name)
    replace_in_file(version_file_path, "REPLACE_BAZEL_BUILD_FLAGS", bazel_bld_flags)

if __name__ == '__main__':
    main()
