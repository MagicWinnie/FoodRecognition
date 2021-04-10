#!/bin/bash
if ! [ -x "$(command -v nvcc)" ]; then
    echo 'CUDA could not be found!' >&2
    exit 1
fi
###
pip install cython pyTelegramBotAPI IPython
###
pip install opencv-python numpy pandas scikit-image matplotlib h5py==2.10.0 imgaug tensorflow-gpu==1.14 keras==2.2.5
###
pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
###
git clone https://github.com/matterport/Mask_RCNN
cd Mask_RCNN/
python setup.py install
###