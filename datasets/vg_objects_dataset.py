#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Containing the building of VGO dataset.
"""
import json, os, pickle
import os.path as osp
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import random
try:
    from .utils import imagenet_preprocess
except:
    from utils import imagenet_preprocess


class object_item(object):
    """
    This is the object item used for object crop retrival
    object_id: The only index linking to object in VG dataset
    object_index: The only index linking to object in COCO dataset
    """
    def __init__(self, object_category=-1, num_objects=None,
                 boxes=None, object_id=None, object_index=None):
        if isinstance(object_category, torch.Tensor):
            object_category = object_category.item()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        self.category = object_category
        x, y, w, h = boxes
        self.x = float(x + w / 2.)
        self.y = float(y + h / 2.)
        self.width = float(w)
        self.height = float(h)
        self.object_id = object_id
        self.object_index = object_index


class Cropped_VG_Dataset(Dataset):
    """Dataset of cropped VG objects."""
    def __init__(self,
                 objects_csv,
                 objects_pickle,
                 mem_bank_path,
                 top10_crop_ids=None,
                 output_size=None,
                 error_handling="None",
                 retrieve_sampling="random",
                 candidate_num=100,
                 normalize_method='imagenet',
        ):
        """
        Return a tensor standardized and reshaped

        Args:
            objects_pickle (string): pickle file with all cropped objects.
            objects_csv: csv containing object information
            output_size: the new width and height of reshaping, it not set, output the original size
            error_handling: what to do with retrieval failure,
                choice 1 -- "None": return None
                choice 2 -- "Zero": return Zero Tensor of (3, ) + output_size specified
        """
        print("Loading objects.csv.")
        self.candidate_num = candidate_num
        self.objects_csv = objects_csv
        self.objects_df = pd.read_csv(objects_csv, dtype={
                "category": np.int32,
                "object_id": np.int32,
                "image_row_number": np.int32,
                "object_idx_in_image": np.int32,
                "image_id": np.int32,
        }, converters={
                "subjects-predicates": eval,
                "objects-predicates": eval,
                "box": eval,
        })
        # initialize the error column in objects_df if it does not exists
        if 'error' not in self.objects_df.columns:
            self.objects_df['error'] = 0

        self.objects_pickle = objects_pickle
        with open(self.objects_pickle, 'rb') as f:
            self.objects = pickle.load(f)

        self.error_handling = error_handling
        self.output_size = output_size
        self.retrieve_sampling = retrieve_sampling
        self.random_flip = False
        self.transform = torch.Tensor

        with open(mem_bank_path) as json_data:
            self.mem_bank = json.load(json_data)

        if retrieve_sampling == "vg_x":
            assert top10_crop_ids, "[top10_crop_ids] is not set."
            with open(top10_crop_ids, "rb") as f:
                self.top10_crop_ids = pickle.load(f)
            if 'val' in osp.basename(top10_crop_ids) or 'test' in osp.basename(top10_crop_ids):
                self.train = False
            else:
                self.train = True

        if retrieve_sampling == "coco_x":
            assert top10_crop_ids, "[top10_crop_ids] is not set."
            with open(top10_crop_ids, "rb") as f:
                self.top10_crop_ids = pickle.load(f)
            if 'val' in osp.basename(top10_crop_ids) or 'test' in osp.basename(top10_crop_ids):
                self.train = False
            else:
                self.train = True

    def __len__(self):
        return len(self.objects_df)

    def __getitem__(self, idx):
        """
        Args:
            idx: object idx as in the new indexing system of objects_df
        """
        key = str(idx)
        if self.objects_pickle:
            np_array = np.array(self.objects[key])
            tensor = self.transform(np_array)
            assert tensor.size(0) != 1
            if self.random_flip and random() > .5:
                tensor = torch.flip(tensor, dims=[2,])
            return tensor


    def retrieve(self, object_item, image_id=None):
        """
        Given the [object_item] containing the basic information
        about the objects to retrive the corresponding object crops
        """
        category = object_item.category
        # retrieve the objects from same category
        same_class_objects = self.mem_bank[str(category)]
        # random select a id in that class,
        # remember random_object is a list containing
        # [image id in h5, object id in image, object id in objects.csv]
        def sample_id(same_class_objects, object_item):
            if self.retrieve_sampling == "random":
                len_class = len(same_class_objects)
                random_object = same_class_objects[np.random.randint(len_class)][-1]
            elif self.retrieve_sampling == "vg_x":
                object_id = object_item.object_id
                selected_ids = self.top10_crop_ids[str(int(object_id))][0]
                if self.train:
                    original_crop = int(selected_ids[0])
                    # selected_crop = int(selected_ids[np.random.choice([0,1,2,3,4,5,6,7,8,9], 1)])
                    selected_crop = int(selected_ids[np.random.choice([0,1,2], 1)])
                    random_object = []
                    random_object.append(selected_crop)
                    random_object.append(original_crop)
                else:
                    original_crop = int(selected_ids[0])
                    selected_crop = int(selected_ids[0])
                    # selected_crop = int(selected_ids[np.random.choice(len(selected_ids), 1)])
                    random_object = []
                    random_object.append(selected_crop)
                    random_object.append(original_crop)
            elif self.retrieve_sampling == "coco_x":
                object_index = object_item.object_index
                selected_ids = self.top10_crop_ids[object_index]
                if self.train:
                    original_crop = int(selected_ids[0])
                    selected_crop = int(selected_ids[np.random.choice([0,1,2], 1)])
                    # selected_crop = int(selected_ids[np.random.choice([0,1,2,3,4,5,6,7,8,9], 1)])
                    random_object = []
                    random_object.append(selected_crop)
                    random_object.append(original_crop)
                else:
                    original_crop = int(selected_ids[0])
                    selected_crop = int(selected_ids[0])
                    # selected_crop = int(selected_ids[np.random.choice(len(selected_ids), 1)])
                    random_object = []
                    random_object.append(selected_crop)
                    random_object.append(original_crop)
            else:
                raise NotImplementedError
            return random_object
        random_id = sample_id(same_class_objects, object_item)
        selected_crops = self[random_id[0]]
        original_crops = self[random_id[1]]

        return selected_crops, original_crops

    def save_objects_csv(self):
        self.objects_df.to_csv(self.objects_csv, index=False)
