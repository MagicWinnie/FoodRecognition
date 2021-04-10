import os
import re
import sys
import math
import time
import random
import argparse

import numpy as np

import tensorflow as tf

from config import CustomConfig, FoodDataset

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.visualize import display_images

parser = argparse.ArgumentParser()
parser.add_argument("--logs",     type=str,
                    default="logs", help="Default log dir")
parser.add_argument("--val",      type=bool, default=True,
                    help="Whether to use photos from validation dataset or own picture")
parser.add_argument("--weights",  type=str,
                    required=True,  help="Weights path")
parser.add_argument("--path",     type=str,  required=True,
                    help="Path to dataset or own picture")
args = parser.parse_args()

WEIGHTS_PATH = args.weights
MODEL_DIR = args.logs

print("[INFO] Initializing configs...")
config_ = CustomConfig()

class InferenceConfig(config_.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 14  # Background + CLASSES
    DETECTION_MIN_CONFIDENCE = 0

    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256

config_ = InferenceConfig()
config_.display()

model = modellib.MaskRCNN(
    mode="inference",
    config=config_,
    model_dir="/content"
)

assert WEIGHTS_PATH != "", "Provide path to trained weights"

print("[INFO] Loading weights from ", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)

print("[INFO] Initializing validation dataset...")
dataset = FoodDataset()
dataset.load_dataset(args.path, load_small=False, return_coco=True)
dataset.prepare()

if args.val:
    fig = plt.figure(figsize=(10, 30))

    for i in range(4):
        image_id = random.choice(dataset.image_ids)

        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(
                dataset,
                config_,
                image_id,
                use_mini_mask=False
            )

        plt.subplot(6, 2, 2*i + 1)

        visualize.display_instances(
            original_image,
            gt_bbox,
            gt_mask,
            gt_class_id,
            dataset.class_names,
            ax=fig.axes[-1]
        )

        plt.subplot(6, 2, 2*i + 2)
        results = model.detect([original_image])  # , verbose=1)
        r = results[0]
        print(r["masks"])
        visualize.display_instances(
            original_image,
            r["rois"],
            r["masks"],
            r["class_ids"],
            dataset.class_names,
            r["scores"],
            ax=fig.axes[-1],
            title="Predictions"
        )
else:
    def get_ax(rows=1, cols=1, size=16):
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    path_to_new_image = args.path
    img = mpimg.imread(path_to_new_image)

    print(len([img]))
    results = model.detect([img], verbose=1)

    ax = get_ax(1)
    r1 = results[0]

    visualize.display_instances(
        img,
        r1["rois"],
        r1["masks"],
        r1["class_ids"],
        dataset.class_names,
        r1["scores"],
        ax=ax,
        title="Predictions"
    )
