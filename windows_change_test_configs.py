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

def help():
    print("Usage:\n\
          windows_change_test_configs.py - replaces default settings \n\
          windows_change_test_configs.py back - replaces default settings back to oryginal\
          ")

def replace_string_in_file(filename, old_string, new_string):
    with open(filename, 'r+', newline='\n') as file:
        filedata = file.read()
        filedata = filedata.replace(old_string, new_string)
        file.seek(0)
        file.write(filedata)
        file.truncate()

def main():
    help()
    replace_back = False
    if len(sys.argv) == 2:
        replace_back = True
        start_dir = os.path.dirname(os.path.realpath(__file__)) + "\\src\\test\\"
        windows_path = start_dir
        print('Setting cwd\\src\\test start search dir: ' + start_dir)
    elif len(sys.argv) == 1:
        start_dir = os.path.dirname(os.path.realpath(__file__)) + "\\src\\test\\"
        windows_path = start_dir
        print('Setting cwd\\src\\test start search dir: ' + start_dir)
    else:
        print("[ERROR] Wrong number of parameters.")
        exit()
            
    linux_path = "/ovms/src/test/"
    
    print("Replacing back set to: " + str(replace_back))

    # Change c:\something\else to c:\\something\\else for json parser compatybility
    windows_path = windows_path.replace("\\","\\\\")
    if replace_back:
        tmp_path = windows_path
        windows_path = linux_path
        linux_path = tmp_path

    print("Replace string: {} with: {} ".format(linux_path, windows_path))

    extension = '.json'  # replace with your desired extension
    files_with_extension = []
    # Use glob to find files with specific extensions
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(extension):
                files_with_extension.append(os.path.join(root, file))

    extension = '.pbtxt'  # replace with your desired extension
    # Use glob to find files with specific extensions
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(extension):
                files_with_extension.append(os.path.join(root, file))

    print("List of files to parse:")
    print("\n".join(map(str, files_with_extension)))

    for file in files_with_extension:
        replace_string_in_file(file, linux_path, windows_path)   

if __name__ == '__main__':
    main()