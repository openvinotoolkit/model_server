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
import tritonclient.grpc as grpcclient
import threading
import time
from PIL import Image
from io import BytesIO

client = grpcclient.InferenceServerClient("localhost:11339")
data = "Zebras in space".encode()

model_name = "python_model"
input_name = "text"

start = time.time()
infer_input = grpcclient.InferInput(input_name, [len(data)], "BYTES")
infer_input._raw_content = data

results = client.infer(model_name, [infer_input])
img = Image.open(BytesIO(results.as_numpy("OUTPUT")))
img.save(f"output.png")
duration = time.time() - start
print(f"Total workers time: {duration} s")