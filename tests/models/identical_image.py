#
# Copyright (c) 2021 Intel Corporation
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

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import numpy as np

model_dir = "/tmp/identity/1"
graph_def = tf.GraphDef()

builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image') # defines input shape
    const = tf.constant(0.0, name="const")
    output = tf.add(input, const, name='output_bytes')  # defines network operation
#    print(sess.graph.get_operations())
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name("image:0")
    out = g.get_tensor_by_name("output_bytes:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"image": inp}, {"output_bytes": out})

    builder.add_meta_graph_and_variables(sess,[tag_constants.SERVING],signature_def_map=sigs)
builder.save()

with tf.Session(graph=tf.Graph()) as sess:
    print("Loading model")
    tf.saved_model.loader.load(sess, ["serve"], model_dir, clear_devices=True)
    print("Model loaded")
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name("image:0")
    preds = graph.get_tensor_by_name("output_bytes:0")

    data = np.ones((1, 224, 224, 3),np.float32)
    feed_dict = {input: data}

    results = sess.run([preds], feed_dict=feed_dict)
    print(results)
