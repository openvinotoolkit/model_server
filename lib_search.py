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

import os
import sys
import re


COPYRIGHT = re.compile(r'Copyright')
INTEL_COPYRIGHT = re.compile(r'Copyright (\(c\) )?(201(8|9)-)?20(20|19|18) Intel Corporation')


def check_header(fd):
    result = False
    detected = False
    try:
        for line in fd:
            if COPYRIGHT.findall(line):
                detected = True
                if INTEL_COPYRIGHT.findall(line):
                    result = True
                    break
    except:
        print("ERROR: Cannot parse file:" + str(fd))
    return detected, result


def check_dir(start_dir):
    ok = []
    not_ok = []
    no_header = []

    exclude_files = ['__pycache__', '.venv', '.pytest_cache', '.vscode', 'ovms-c/dist', '.git', '.tar.gz', 'docx',
                     '.npy', '.png', '.svg', '.bin', '.jpeg', '.jpg', 'license.txt', 'md', '.groovy', '.json' ,'bazel-',
                     'Doxyfile', 'clang-format','net_http.patch', 'tftext.patch', 'tf.patch', 'client_requirements.txt',
                     'openvino.LICENSE.txt', 'c-ares.LICENSE.txt', 'zlib.LICENSE.txt', 'boost.LICENSE.txt',
                     'libuuid.LICENSE.txt', 'input_images.txt', 'REST_age_gender.ipynb', 'dummy.xml', 'listen.patch', 'add.xml',
                     'requirements.txt', 'missing_headers.txt', 'libevent/BUILD', 'azure_sdk.patch', 'rest_sdk_v2.10.16.patch',]
                   
    exclude_directories = ['/dist/']

    for (d_path, dir_set, file_set) in os.walk(start_dir):
        for f_name in file_set:
            
            skip = False
            for excluded in exclude_directories:
                if excluded in d_path:
                    skip = True
                    print('Warning - Skipping directory - ' + d_path + ' for file - ' + f_name)
                    break
                
            if skip:
                continue

            fpath = os.path.join(d_path, f_name)

            if not [test for test in exclude_files if test in fpath]:
                with open(fpath, 'r') as fd:
                    header_detected, result = check_header(fd)
                    if header_detected:
                        if result:
                            ok.append(fpath)
                        else:
                            not_ok.append(fpath)
                    else:
                        no_header.append(fpath)
    return not_ok, no_header


def main():
    if len(sys.argv) < 1:
        print('Provide start dir!')
    else:
        start_dir = sys.argv[1]

        external_component_set, no_header_set = check_dir(start_dir)

        if len(no_header_set) == 0:
            print('Success: All files have headers')
        else:
            print('#########################')
            print('## No header files detected:')
            for no_header in no_header_set:
                print(f'{no_header}')

    
if __name__ == '__main__':
    main()
