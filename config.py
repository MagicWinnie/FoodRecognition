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
    NUM_CLASSES = 1 + 14  # Background + CLASSES
    STEPS_PER_EPOCH = 150
    VALIDATION_STEPS = 50

    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256

class FoodDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, return_coco=True):
        """ Loads dataset
            Params:
                - dataset_dir : root directory of the dataset (can point to the train/val folder)
                - load_small : Boolean value which signals if the annotations for all the images need to be loaded into the memory,
                               or if only a small subset of the same should be loaded into memory
        """
        annotation_path = os.path.join(dataset_dir, "annotations.json")

        image_dir = os.path.join(dataset_dir, "images")
        print("[INFO] Annotation Path ", annotation_path)
        print("[INFO] Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes (Only Building in this version)
        classIds = self.coco.getCatIds()
        
        # Load all images
        image_ids = list(self.coco.imgs.keys())
        
        # register classes
        for _class_id in classIds:
            self.add_class(name, _class_id,
                           self.coco.loadCats(_class_id)[0]["name"])

        # Register Images
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
        """ Loads instance mask for a given image
              This function converts mask from the coco format to a
              a bitmap [height, width, instance]
            Params:
                - image_id : reference id for a given image

            Returns:
                masks : A bool array of shape [height, width, instances] with
                    one mask per instance
                class_ids : a 1D array of classIds of the corresponding instance masks
                    (In this version of the challenge it will be of shape [instances] and always be filled with the class-id of the "Building" class.)
        """
        image_info = self.image_info[image_id]
        assert image_info["source"] == name

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                f"{name}.{annotation['category_id']}"
            )
            if class_id:
                m = self.annToMask(annotation,
                                   image_info["height"],
                                   image_info["width"]
                                   )

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset

                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(FoodDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """ Return a reference for a particular image

            Ideally you this function is supposed to return a URL
            but in this case, we will simply return the image_id
        """
        return f"{name}::{image_id}"
    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """ Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann, height, width):
        """ Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
