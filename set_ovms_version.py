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

WIN_OV_VERSION_REGEX = re.compile(r'[0-9]{4}.[0-9].[0-9].[^_]+')
VERSION_FILE = "src\\version.hpp"
OVMS_PROJECT_VERSION="2025.0"

def help():
    print("Usage:\n\
          Two arguments required: BAZEL_BUILD_FLAGS PATH_TO_OPENVINO_INSTALL\n\
          for example: \n\
          set_ovms_version.py.\"--config=windows\" c:\opt\ \n\
          ")
    
def replace_in_file(file_path, old_string, new_string):
    with open(file_path, 'r+') as file:
        contents = file.read()
        contents = contents.replace(old_string, new_string)
        file.seek(0)
        file.write(contents)
        file.truncate()

def get_openvino_name(openvino_dir):
    openvino_name = "Unknown"
    for root, dirs, files in os.walk(openvino_dir):
        for dir in dirs:
            print(dir)
            matches = WIN_OV_VERSION_REGEX.findall(dir)
            if len(matches) > 1:
                print("[ERROR] Multiple openvino versions detected in " + os.path.join(root, dir))
                exit(-1)
            if len(matches) == 1:
                print("[INFO] Openvino detected in " + os.path.join(root, dir))
                openvino_name = matches[0]
                break
        
        # we search only 1 directory level deep
        break

    if openvino_name == "Unknown":
        print("[WARNING] Openvino versions not detected in " + openvino_dir)

    return openvino_name

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

    openvino_name = get_openvino_name(openvino_dir)
    version_file_path = os.path.join(os.getcwd(), VERSION_FILE)
    replace_in_file(version_file_path, "REPLACE_PROJECT_VERSION", OVMS_PROJECT_VERSION)
    replace_in_file(version_file_path, "REPLACE_OPENVINO_NAME", openvino_name)
    replace_in_file(version_file_path, "REPLACE_BAZEL_BUILD_FLAGS", bazel_bld_flags)

if __name__ == '__main__':
    main()
