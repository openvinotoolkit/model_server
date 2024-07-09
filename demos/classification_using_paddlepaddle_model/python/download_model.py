#
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
import hashlib
import os.path
import urllib.request
import tarfile

mobilenet_url="https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar"
mobilenetv3_model_path = "model/MobileNetV3_large_x1_0_infer/inference.pdmodel"
if os.path.isfile(mobilenetv3_model_path): 
    print("Model MobileNetV3_large_x1_0 already existed")
else:
    #Download the model from the server, and untar it.
    print("Downloading the MobileNetV3_large_x1_0_infer model (20Mb)... May take a while...")
    #make the directory if it is not 
    os.makedirs('model')
    urllib.request.urlretrieve(mobilenet_url, "model/MobileNetV3_large_x1_0_infer.tar")  # nosec
    print("Model Downloaded")

    sha256_hash = hashlib.sha256()
    with open("model/MobileNetV3_large_x1_0_infer.tar", "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    calculated_hash = sha256_hash.hexdigest()
    expected_hash = "ca6c1b685b40c8d9219563f52b76c5716a2fbdc5214ebcdba67b234db8070f72"
    if calculated_hash != expected_hash:
        print("Downloaded file integrity check failed")
        exit(1)
    print("File integrity verified. Extracting the tar archive...")

    file = tarfile.open("model/MobileNetV3_large_x1_0_infer.tar")
    res = file.extractall('model')  # nosec
    file.close()
    if (not res):
        print("Model Extracted to \"model/MobileNetV3_large_x1_0_infer\".")
    else:
        print("Error Extracting the model. Please check the network.")

    os.rename('model/MobileNetV3_large_x1_0_infer', 'model/1')
    os.remove('model/MobileNetV3_large_x1_0_infer.tar')
    print("Workspace created")

