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

import argparse
import datetime
import numpy as np
from client_utils import print_statistics
from urllib.request import urlretrieve
from pathlib import Path
import os
from PIL import Image
import openvino as ov
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from scipy.special import softmax

parser = argparse.ArgumentParser(description='Client for clip example')

parser.add_argument('--url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000')
parser.add_argument('--input_labels', required=False, default="cat,dog,wolf,tiger,man,horse,frog,tree,house,computer",
                    help="Specify input_labels to the CLIP model. default:cat,dog,wolf,tiger,man,horse,frog,tree,house,computer")
parser.add_argument('--image_url', required=False, default='https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg',
                    help='Specify image_url to send to the CLIP model. default:https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg')
parser.add_argument('--iterations', default=1,
                        help='Number of requests iterations, as default use number of images in numpy memmap. default: 1 ',
                        dest='iterations', type=int)
args = vars(parser.parse_args())

iterations = args.get('iterations')
iteration = 0

image_url = args['image_url']
print(f"Using image_url:\n{image_url}\n")

input_name = image_url.split("/")[-1]
sample_path = Path(os.path.join("data", input_name))
if not os.path.exists(sample_path):
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        image_url,
        sample_path,
    )
#image = Image.open(sample_path)
with open(sample_path, "rb") as f:
    data = f.read()

# print(f"Using input_labels:\n{input_labels}\n")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# load preprocessor for model input
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

core = ov.Core()
ov_model = core.read_model("/model/1/clip-vit-base-patch16.xml")
compiled_model = core.compile_model(ov_model, "CPU")
logits_per_image_out = compiled_model.output(0)

processing_times = np.zeros((0),int)
while iteration < iterations:
    iteration += 1
    print(f"Iteration {iteration}")
    start_time = datetime.datetime.now()
    # TIME START

    input_labels = args['input_labels'].split(",")
    image = Image.open(BytesIO(data))
    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]
    inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)

    ov_logits_per_image = compiled_model(dict(inputs))[logits_per_image_out]
    
    probs = softmax(ov_logits_per_image, axis=1)
    max_prob = probs[0].argmax(axis=0)
    max_label = input_labels[max_prob]
    max_label = str(max_label)

    # TIME STOP
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    processing_times = np.append(processing_times,np.array([int(duration)]))
    print(f"Detection:\n{max_label}\n")

batch_size = 1
print_statistics(processing_times, batch_size)