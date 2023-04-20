import os
import re

base_path = "ovmsclient/tfs_compat/protos"

# Find all proto files
proto_paths = []
for root, subdirs, files in os.walk(base_path):
    for file in files:
        if file[-6:] == ".proto":
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