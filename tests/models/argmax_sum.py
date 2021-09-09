#
# Copyright (c) 2020 Intel Corporation
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

import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='generate model calculation argmax from 2 inputs')
parser.add_argument('--input_size', type=int, help='input tensor size', dest='input_size', default=10)
parser.add_argument('--export_dir', help='directory to store the model', dest='export_dir', default="/tmp/argmax/1")
args = vars(parser.parse_args())

print(args.get('input_size'))
model_dir = args.get('export_dir')
graph_def = tf.GraphDef()

builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    input1 = tf.placeholder(tf.float32, [None, args.get('input_size')], name='input1') # defines input shape
    input2 = tf.placeholder(tf.float32, [None, args.get('input_size')], name='input2') # defines input shape
    sum = tf.add(input1, input2, name='sum')  # add 2 inputs
    max = tf.argmax(sum,1,name='argmax')  # get index with max value
    g = tf.get_default_graph()
    inp1 = g.get_tensor_by_name("input1:0")
    inp2 = g.get_tensor_by_name("input2:0")
    m = g.get_tensor_by_name("argmax:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in1": inp1, "in2": inp2}, {"argmax":m})
    builder.add_meta_graph_and_variables(sess,[tag_constants.SERVING],signature_def_map=sigs)
builder.save()

# evaluation
with tf.Session(graph=tf.Graph()) as sess:
    print("Loading model")
    tf.saved_model.loader.load(sess, ["serve"], model_dir, clear_devices=True)
    print("Model loaded")
    graph = tf.get_default_graph()
    input1 = graph.get_tensor_by_name("input1:0")
    input2 = graph.get_tensor_by_name("input2:0")
    # preds = graph.get_tensor_by_name("output:0")
    m = graph.get_tensor_by_name("argmax:0")
    data1 = np.ones((1,args.get('input_size')),np.float32)
    data2 = np.ones((1,args.get('input_size')),np.float32)
    data2[0,3] = 5
    feed_dict = {input1: data1, input2: data2}
    results = sess.run([m], feed_dict=feed_dict)
    print(results)
