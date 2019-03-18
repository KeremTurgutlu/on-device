import onnx
import onnx_tf
from onnx_tf.backend import prepare
import os

# models dir
models_dir = "pytorch_conversions"
outputs_dir = "pytorch_conversions/outputs"
DIRS = [models_dir, outputs_dir]
for DIR in DIRS: os.makedirs(DIR, exist_ok=True)

onnx_filename = models_dir + "/resnet18.onnx"
tf_proto_filename = models_dir + "/resnet18.pb"

onnx_model = onnx.load(onnx_filename)
tf_rep = prepare(onnx_model, strict=False) # strict=True fails 
tf_rep.export_graph(tf_proto_filename)  # export the model

print("input and output tensor names:")
print(f"inputs: {tf_rep.inputs}")
print(f"inputs: {tf_rep.outputs}")
