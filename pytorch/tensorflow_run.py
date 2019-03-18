import tensorflow as tf
import numpy as np

input_image = np.load("input_image.npy")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.ConfigProto(gpu_options=gpu_options)

graph_def = tf.GraphDef()
with open("resnet18.pb", "rb") as f:
    graph_def.ParseFromString(f.read())
    
with tf.device("/device:GPU:0"):
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
        
    with tf.Session(graph=graph, config=config) as sess:   
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        input_node = graph.get_tensor_by_name('prefix/0:0')
        output_node = graph.get_tensor_by_name('prefix/add_9:0')
        feed_dict={input_node: input_image}
        tf_output = sess.run(output_node, feed_dict)    
        print(tf_output)
    


# # this works
# with tf.device("/device:GPU:0"):
#     # Using `Tensor.eval()`.
#     c = tf.constant(5.0)
#     with tf.Session(config=config):
#         print(c.eval())