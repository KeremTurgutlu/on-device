# Conversion Environment Dependencies

# conda env
conda create --name conversion python=3.6
source activate conversion

# fastai - dev, pytorch - '1.0.1.post2', torchvision - '0.2.1'
conda install -c pytorch -c fastai fastai
conda uninstall --force jpeg libtiff -y
conda install -c conda-forge libjpeg-turbo
CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall --no-binary :all: --compile pillow-simd
conda install jupyter notebook
conda install -c conda-forge jupyter_contrib_nbextensions
conda install nb_conda

# tensorflow - '1.13.1'
pip install tensorflow-gpu

# onnx - '1.4.1'
conda install -c conda-forge onnx 

# onnx-coreml - latest, doesn't work with python=3.7
pip install -U onnx-coreml

# onnx-tf -latest, because pip install onnx-tf - batchnorm v9 not supported
git clone git@github.com:onnx/onnx-tensorflow.git && cd onnx-tensorflow
pip install -e .

# needed for caffe2
pip install future


###########
### KERAS ###
###########

# for keras to onnx
pip install onnxmltools

# tf 2 coreml
pip install -U tfcoreml











