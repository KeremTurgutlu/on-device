# pytorch = '1.0.1.post2'
import warnings; warnings.filterwarnings("ignore")
import torch
import torchvision
import numpy as np
import os

# models dir
models_dir = "pytorch_conversions"
outputs_dir = "pytorch_conversions/outputs"
DIRS = [models_dir, outputs_dir]
for DIR in DIRS: os.makedirs(DIR, exist_ok=True)

# input variables
onnx_filename = models_dir + "/resnet18.onnx"
input_image = np.load('input_image.npy')
torch_input = torch.tensor(input_image)

# load model
model = torchvision.models.resnet18(True)

# covnvert to onnx using torch.onnx
torch.onnx.export(model, torch_input,
                  onnx_filename,
                  verbose=True)

torch_output = model(torch_input)
torch_output_np = torch_output.detach().numpy()
np.save(outputs_dir+"/torch_output.npy", torch_output_np)


