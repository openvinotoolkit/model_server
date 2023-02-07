#No header files detected
# Copyright (c) 2020-2022 Intel Corporation
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

COPYRIGHT = re.compile(r'(C|c)opyright (\((c|C)\) )?(20(0|1|2)([0-9])-)?(20(0|1|2)([0-9]),)*20(0|1|2)([0-9])(,)? ([a-z]|[A-Z]|[0-9]| )*')
FORBIDDEN_FUNCTIONS = re.compile(r'setjmp\(|longjmp\(|getwd\(|strlen\(|wcslen\(|gets\(|strcpy\(|wcscpy\(|strcat\(|wcscat\(|sprintf\(|vsprintf\(|asctime\(')


def check_header(fd):
    detected = False
    try:
        for line in fd:
            if COPYRIGHT.findall(line):
                detected = True
                break
    except:
        print("ERROR: Cannot parse file:" + str(fd))
    return detected

def check_function(fd):
    # Add space separated exceptions for given file in the dictionary
    fix_applied = {"./src/test/ensemble_flow_custom_node_tests.cpp":"size_t strLen = std::strlen(str);size_t prefixLen = std::strlen(prefix);",}

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
    no_header = []
    exclude_files = [
        '.bandit',
        '.bin',
        '.git',
        '.groovy',
        '.jpeg',
        '.jpg',
        '.json',
        '.md',
        '.npy',
        '.pdf',
        '.png',
        '.proto',
        '.pytest_cache',
        '.svg',
        '.tar.gz',
        '.venv',
        '.wav',
        'Doxyfile',
        'LICENSE',
        'REST_age_gender.ipynb',
        '__pycache__',
        'add.xml',
        'azure_sdk.patch',
        'bazel-',
        'check_coverage.bat',
        'genhtml',
        'clang-format',
        'client_requirements.txt',
        'cppclean_src',
        'cppclean_test',
        'docx',
        'dummy.xml',
        'passthrough.xml',
        'forbidden_functions.txt',
        'gif',
        'increment_1x3x4x5.xml',
        'index.html',
        'input_images.txt',
        'libevent/BUILD',
        'listen.patch',
        'metrics_output.out',
        'missing_headers.txt',
        'model.xml',
        'net_http.patch',
        'opencv_cmake_flags.txt',
        'ovms-c/dist',
        'requirements.txt',
        'resnet_images.txt',
        "resnet_labels.txt",
        'rest_sdk_v2.10.16.patch',
        'summator.xml',
        'tf.patch',
        'tftext.patch',
        'vehicle_images.txt',
        'bazel_rules_apple.patch',
        "pom.xml",
        "go.sum",
        "mwaitpkg.patch",
        'saved_model.pb',
        "yarn.lock",
        "BUILD.bazel",
        "package.json",
        ]

    exclude_directories = ['/dist/', 'extras/ovms-operator', 'extras/openvino-operator-openshift', 'release_files/thirdparty-licenses']

    for (d_path, _, file_set) in os.walk(start_dir):
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
                    header_detected = check_header(fd)
                    if not header_detected:
                        no_header.append(fpath)
    return no_header

def check_func(start_dir):
    not_ok = []

    exclude_files = [
        '.bin',
        '.git',
        '.groovy',
        '.jpeg',
        '.jpg',
        '.json' ,
        '.npy',
        '.png',
        '.pytest_cache',
        '.svg',
        '.tar.gz',
        '.venv',
        '.vscode',
        'Doxyfile',
        'REST_age_gender.ipynb',
        '__pycache__',
        'add.xml',
        'azure_sdk.patch',
        'bazel-',
        'boost.LICENSE.txt',
        'c-ares.LICENSE.txt',
        'check_coverage.bat',
        'genhtml',
        'clang-format',
        'client_requirements.txt',
        'docx',
        'dummy.xml',
        'passthrough.xml',
        'forbidden_functions.txt',
        'input_images.txt',
        'libevent/BUILD',
        'libuuid.LICENSE.txt',
        'license.txt',
        'listen.patch',
        'md',
        'metrics_output.out',
        'missing_headers.txt',
        'missing_headers.txt',
        'net_http.patch',
        'openvino.LICENSE.txt',
        'ovms-c/dist',
        'requirements.txt',
        'rest_sdk_v2.10.16.patch',
        'summator.xml',
        'tf.patch',
        'tftext.patch',
        'zlib.LICENSE.txt',
        'bazel_rules_apple.patch',
        'yarn.lock',
        'BUILD.bazel',
        'package.json',
    ]

    exclude_directories = ['/dist/', 'extras/ovms-operator']

    for (d_path, _, file_set) in os.walk(start_dir):
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
        no_header_set = check_dir(start_dir)

        if len(no_header_set) == 0:
            print('Success: All files have headers')
        else:
            print('#########################')
            print('## No header files detected:')
            for no_header in no_header_set:
                print(f'{no_header}')


if __name__ == '__main__':
    main()
