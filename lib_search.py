#
# Copyright (c) 2020-2021 Intel Corporation
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
FORBIDDEN_FUNCTIONS = re.compile(r'setjmp\(|longjmp\(|getwd\(|strlen\(|wcslen\(|gets\(|strcpy\(|wcscpy\(|strcat\(|wcscat\(|sprintf\(|vsprintf\(|asctime\(')


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

def check_function(fd):
    # Add space separated exceptions for given file in the dictionary
    fix_applied = {"./src/test/ensemble_flow_custom_node_tests.cpp":"size_t strLen = std::strlen(str);size_t prefixLen = std::strlen(prefix);",}

    result = False
    detected = False
    try:
        for line in fd:
            found = FORBIDDEN_FUNCTIONS.findall(line)
            if found:
                if line.trim() in fix_applied[fd.name]:
                    #It's ok fix and check was applied and verified
                    continue
                detected = True
                print("ERROR: Forbidden function detected in:" + str(fd.name))
                print("Line start:" + str(line) + "End")
                print("Function:" + str(found))
                break
    except:
        print("ERROR: Cannot parse file:" + str(fd))
    return detected


def check_dir(start_dir):
    ok = []
    not_ok = []
    no_header = []

    exclude_files = ['__pycache__', '.venv', '.pytest_cache', '.vscode', 'ovms-c/dist', '.git', '.tar.gz', 'docx',
                     '.npy', '.png', '.svg', '.bin', '.jpeg', '.jpg', 'license.txt', 'md', '.groovy', '.json' ,'bazel-',
                     'Doxyfile', 'clang-format','net_http.patch', 'tftext.patch', 'tf.patch', 'client_requirements.txt',
                     'openvino.LICENSE.txt', 'c-ares.LICENSE.txt', 'zlib.LICENSE.txt', 'boost.LICENSE.txt',
                     'libuuid.LICENSE.txt', 'input_images.txt', 'REST_age_gender.ipynb', 'dummy.xml', 'listen.patch', 'add.xml',
                     'requirements.txt', 'missing_headers.txt', 'libevent/BUILD', 'azure_sdk.patch', 'rest_sdk_v2.10.16.patch', '.wav',
                     'forbidden_functions.txt', 'missing_headers.txt', 'increment_1x3x4x5.xml', 'horizontal-text-detection.gif', 'model.xml']

    exclude_directories = ['/dist/', 'extras/ovms-operator', 'extras/openvino-operator-openshift']

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

def check_func(start_dir):
    ok = []
    not_ok = []

    exclude_files = ['__pycache__', '.venv', '.pytest_cache', '.vscode', 'ovms-c/dist', '.git', '.tar.gz', 'docx',
                     '.npy', '.png', '.svg', '.bin', '.jpeg', '.jpg', 'license.txt', 'md', '.groovy', '.json' ,'bazel-',
                     'Doxyfile', 'clang-format','net_http.patch', 'tftext.patch', 'tf.patch', 'client_requirements.txt',
                     'openvino.LICENSE.txt', 'c-ares.LICENSE.txt', 'zlib.LICENSE.txt', 'boost.LICENSE.txt',
                     'libuuid.LICENSE.txt', 'input_images.txt', 'REST_age_gender.ipynb', 'dummy.xml', 'listen.patch', 'add.xml',
                     'requirements.txt', 'missing_headers.txt', 'libevent/BUILD', 'azure_sdk.patch', 'rest_sdk_v2.10.16.patch', 'forbidden_functions.txt', 'missing_headers.txt']

    exclude_directories = ['/dist/', 'extras/ovms-operator']

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
                    detected = check_function(fd)
                    if detected:
                        not_ok.append(fpath)
                    else:
                        ok.append(fpath)
    return not_ok

def main():
    if len(sys.argv) < 2:
        print('Provide start dir!')
    else:
        start_dir = sys.argv[1]
        print('Provided start dir:' + start_dir)


    if len(sys.argv) > 2 and sys.argv[2] == 'functions':
        print("Check for forbidden functions")
        forbidden_func = check_func(start_dir)

        if len(forbidden_func) == 0:
            print('Success: All files checked for forbidden functions')
        else:
            print('#########################')
            print('## Forbidden functions detected:')
            for forbid_func in forbidden_func:
                print(f'{forbid_func}')


    else:
        print("Check for missing headers")
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
