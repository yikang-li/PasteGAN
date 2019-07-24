#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Containing the building of COCO Objects Dataset.
"""

import json, os
import numpy as np
import pandas as pd
import h5py
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .vg_objects_dataset import object_item, Cropped_VG_Dataset
except:
    from vg_objects_dataset import object_item, Cropped_VG_Dataset

class Cropped_COCO_Dataset(Cropped_VG_Dataset):
    def __init__(self,
                 objects_csv,
                 objects_pickle,
                 mem_bank_path,
                 top10_crop_ids=None,
                 output_size=None,
                 error_handling="None",
                 retrieve_sampling="ratio",
                 candidate_num=100,
                 normalize_method='imagenet',
        ):
        # Inherit the Cropped_VG_Dataset
        super(Cropped_COCO_Dataset, self).__init__(objects_csv,
                                                   objects_pickle,
                                                   mem_bank_path,
                                                   top10_crop_ids,
                                                   output_size,
                                                   error_handling,
                                                   retrieve_sampling,
                                                   candidate_num,
                                                   normalize_method,)
        self.objects_pickle = objects_pickle
        if self.objects_pickle is not None:
            with open(self.objects_pickle, 'rb') as f:
                self.objects = pickle.load(f)
        self.transform = torch.Tensor

    def __getitem__(self, idx):
        """
        Args:
            idx: object idx as in the new indexing system of objects_df
        Output:
            tensor: torch tensor of image crop
        """
        np_array = np.array(self.objects[str(idx)])
        tensor = self.transform(np_array)
        return tensor


    def get_mask(self, idx):
        """
        Args:
            idx: object idx as in the new indexing system of objects_df
        """
        assert not self.features
        np_array = self.masks[idx].astype(np.int32)
        tensor = self.transform(np_array)
        return tensor
