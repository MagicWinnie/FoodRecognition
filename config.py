import os
import json

import numpy as np

import mrcnn
import mrcnn.utils
import mrcnn.config
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import model as modellib, utils
from mrcnn.visualize import display_instances

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

name = "foodrec"


class CustomConfig(Config):
    NAME = name
    BACKBONE = "resnet50"

    LEARNING_RATE = 0.001
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 14  # background + classes
    STEPS_PER_EPOCH = 150
    VALIDATION_STEPS = 50

    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256

class FoodDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, return_coco=True):
        annotation_path = os.path.join(dataset_dir, "annotations.json")

        image_dir = os.path.join(dataset_dir, "images")
        print("[INFO] Annotation Path ", annotation_path)
        print("[INFO] Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        classIds = self.coco.getCatIds()
        
        image_ids = list(self.coco.imgs.keys())
        
        for _class_id in classIds:
            self.add_class(name, _class_id,
                           self.coco.loadCats(_class_id)[0]["name"])

        ctgs = []
        for _img_id in image_ids:
            assert(os.path.exists(os.path.join(
                image_dir, self.coco.imgs[_img_id]["file_name"])))
            self.add_image(
                name,
                image_id=_img_id,
                path=os.path.join(
                    image_dir, self.coco.imgs[_img_id]["file_name"]),
                width=self.coco.imgs[_img_id]["width"],
                height=self.coco.imgs[_img_id]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[_img_id],
                    catIds=classIds,
                    iscrowd=None
                )
                )
            )
            temp = self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[_img_id],
                    catIds=classIds,
                    iscrowd=None
                )
                )
            for i in range(len(temp)):
                ctgs.append(temp[i]['category_id'])
            # print(temp)
        print(list(set(ctgs)))

        if return_coco:
            return self.coco

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        assert image_info["source"] == name

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                f"{name}.{annotation['category_id']}"
            )
            if class_id:
                m = self.annToMask(annotation,
                                   image_info["height"],
                                   image_info["width"]
                                   )

                if m.max() < 1:
                    continue

                instance_masks.append(m)
                class_ids.append(class_id)
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(FoodDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        return f"{name}::{image_id}"

    def annToRLE(self, ann, height, width):
        segm = ann["segmentation"]
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm["counts"], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
