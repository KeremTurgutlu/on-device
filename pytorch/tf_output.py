import warnings; warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import os

# GPU config
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
config.gpu_options.allow_growth = True

# models dir
models_dir = "pytorch_conversions"
outputs_dir = "pytorch_conversions/outputs"
DIRS = [models_dir, outputs_dir]
for DIR in DIRS: os.makedirs(DIR, exist_ok=True)

input_image = np.load('input_image.npy')
tf_proto_filename = models_dir + "/resnet18.pb"

with tf.device("/device:GPU:0"):
    graph_def = tf.GraphDef()        
    with open(tf_proto_filename, "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()

        input_x = sess.graph.get_tensor_by_name("0:0") # input name
        output = sess.graph.get_tensor_by_name('add_9:0') # output name

        print(input_x, output)
        tf_output = sess.run([output], feed_dict={input_x:input_image})
        print('tf_output = {}'.format(tf_output))

np.save(outputs_dir+"/tf_output.npy", tf_output)