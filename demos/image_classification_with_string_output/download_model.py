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


import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import shutil
import requests
import numpy as np
import tensorflow as tf
import keras

def preprocess(image, mean=0.5, std=0.5, shape=(224, 224)):
    """Scale, normalize and resizes images."""
    image = image / 255.0
    image = (image - mean) / std
    image = tf.image.resize(image, shape)
    return image

imagenet_labels_url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
)
response = requests.get(imagenet_labels_url)
labels = [x for x in response.text.split("\n") if x != ""][1:]
tf_labels = tf.constant(labels, dtype=tf.string)


def postprocess(prediction, labels=tf_labels):
    """Convert from probs to labels."""
    indices = tf.argmax(prediction, axis=-1)
    label = tf.gather(params=labels, indices=indices)
    return label

def export_model(model, labels):
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def serving_fn(image):
        processed_img = preprocess(image)
        probs = model(processed_img)
        label = postprocess(probs)
        return {"label": label}

    return serving_fn

model = keras.applications.MobileNet()
model_dir = "./model"
model_version = 1
model_export_path = f"{model_dir}/{model_version}"

tf.saved_model.save(
    model,
    export_dir=model_export_path,
    signatures={"serving_default": export_model(model, labels)},
)


 