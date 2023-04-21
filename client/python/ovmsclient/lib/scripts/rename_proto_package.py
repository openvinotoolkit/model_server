#
# Copyright (c) 2023 Intel Corporation
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

base_path = "ovmsclient/tfs_compat/protos"

# Find all proto files
proto_paths = []
for root, subdirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".proto"):
            file_path = os.path.join(root, file)
            proto_paths.append(file_path)

# Replace package name if defined in all proto files
replacement_map = {
    "package tensorflow": "package ovmsclient",
    " tensorflow.": " ovmsclient.",
    " .tensorflow.": " .ovmsclient."
}

for proto_path in proto_paths:
    with open(proto_path, 'rt') as file :
        filedata = file.read()

    for to_replace, replace_with in replacement_map.items():
        filedata = filedata.replace(to_replace, replace_with)

    with open(proto_path, 'wt') as file:
        file.write(filedata)
