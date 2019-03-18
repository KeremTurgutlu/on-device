import warnings; warnings.filterwarnings("ignore")
import torch
import torchvision
import tensorflow as tf
import numpy as np
import onnx
import onnx_tf
from onnx_tf.backend import prepare
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

model = torchvision.models.resnet18(True)

dummy_input = torch.randn(1, 3, 224, 224) # nchw

onnx_filename = "resnet18.onnx"

torch.onnx.export(model, dummy_input,
                  onnx_filename,
                  verbose=True)

torch_output = model(dummy_input)

torch_output_np = torch_output.detach().numpy()

torch_output_np.shape

dummy_input_np = dummy_input.numpy()

dummy_input_np.shape

# step 2, create onnx_model using tensorflow as backend. check if right and export graph.
onnx_model = onnx.load(onnx_filename)
tf_rep = prepare(onnx_model, strict=False)

tf_rep.tensor_dict[tf_rep.inputs[0]], tf_rep.tensor_dict[tf_rep.outputs[0]]

with tf.device("/device:CPU:0"):
    with tf_rep.graph.as_default():
        with tf.Session() as sess:
            feed_dict = {tf_rep.tensor_dict[key]: dummy_input_np for key in tf_rep.inputs}
            sess.run(tf.global_variables_initializer())
            outputs = [tf_rep.tensor_dict[output] for output in tf_rep.outputs]
            tf_output = sess.run(outputs, feed_dict=feed_dict)
 
# print(torch_output_np, tf_output)
input_image = np.load('input_image.npy')  

with tf.device("/device:GPU:0"):
    graph_def = tf.GraphDef()        
    with open("./resnet18.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()

        input_x = sess.graph.get_tensor_by_name("0:0") # input
        output = sess.graph.get_tensor_by_name('add_9:0') # 5

        print(input_x, output)
        output_tf_pb = sess.run([output], feed_dict={input_x:input_image})
        print('output_tf_pb = {}'.format(output_tf_pb))

# this works
# with tf.device("/device:CPU:0"):
#     # Using `Tensor.eval()`.
#     c = tf.constant(5.0)
#     with tf.Session(config=config):
#         print(c.eval())













