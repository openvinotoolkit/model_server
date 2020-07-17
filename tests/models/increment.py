import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import numpy as np

model_dir = "/tmp/increment/1"
graph_def = tf.GraphDef()

builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    input = tf.placeholder(tf.float32, [None, 10], name='input') # defines input shape
    const = tf.constant(1.0, name="const")
    output = tf.add(input, const, name='output')  # defines network operation
#    print(sess.graph.get_operations())
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name("input:0")
    out = g.get_tensor_by_name("output:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess,[tag_constants.SERVING],signature_def_map=sigs)
builder.save()

with tf.Session(graph=tf.Graph()) as sess:
    print("Loading model")
    tf.saved_model.loader.load(sess, ["serve"], model_dir, clear_devices=True)
    print("Model loaded")
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name("input:0")
    preds = graph.get_tensor_by_name("output:0")

    data = np.ones((1,10),np.float32)
    feed_dict = {input: data}

    results = sess.run([preds], feed_dict=feed_dict)
    print(results)
