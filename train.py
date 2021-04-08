import os
import argparse
import numpy as np
import keras.backend

from config import CustomConfig, CustomDataset

import mrcnn
import mrcnn.utils
from mrcnn.config    import Config
from mrcnn.model     import MaskRCNN
from mrcnn           import model as modellib

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, required=True, help="Train dataset path")
parser.add_argument("--val", type=str, required=True, help="Validation dataset path")
parser.add_argument("--logs", type=str, default="logs", help="Default log dir")
parser.add_argument("--coco", type=str, default="mask1_rcnn_coco.h5", help="COCO weights path")
parser.add_argument("--layers", type=str, default="heads", help="Which layers to train (heads, all)")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to learn")
args = parser.parse_args()

DEFAULT_LOGS_DIR = args.logs
COCO_WEIGHTS_PATH = args.coco

print("[INFO] Downloading COCO weights...")
if not os.path.exists(COCO_WEIGHTS_PATH):
    mrcnn.utils.download_trained_weights(COCO_WEIGHTS_PATH)

print("[INFO] Initializing configs...")
config_ = CustomConfig()
K = keras.backend.backend()
keras.backend.common.image_dim_ordering()
model = modellib.MaskRCNN(
    mode      = "training",
    config    = config_,
    model_dir = DEFAULT_LOGS_DIR
)

print("[INFO] Loading COCO weights...")
model.load_weights(
    COCO_WEIGHTS_PATH,
    by_name = True,
    exclude = [
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"
    ]
)

print("[INFO] Initializing train dataset...")
dataset_train = CustomDataset()
dataset_train.load_dataset(args.train, load_small = False)
dataset_train.prepare()

print("[INFO] Initializing validation dataset...")
dataset_val = CustomDataset()
dataset_val.load_dataset(args.val, load_small = False, return_coco=True)
dataset_val.prepare()

print("[INFO] Starting training...")
model.train(
    dataset_train, dataset_val,
    learning_rate = config_.LEARNING_RATE,
    epochs        = args.epochs,
    layers        = args.layers
)
