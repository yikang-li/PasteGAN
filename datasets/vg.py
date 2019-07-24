import sys
import os, json
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL

from .utils import imagenet_preprocess
from .vg_objects_dataset import Cropped_VG_Dataset, object_item
from utils.canvas import make_canvas_baseline

import warnings
# This ignore all the warning because of skimage
warnings.filterwarnings('ignore')


class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
                 max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True,
                 use_object_crops=True, normalize_method='imagenet', **kwargs):
        super(VgSceneGraphDataset, self).__init__()

        # define the location of the directory
        self.image_dir = image_dir
        # should be the output image size
        self.image_size = image_size
        # vocabulary
        self.vocab = vocab
        # total number of objects
        self.num_objects = len(vocab['object_idx_to_name'])
        # I guess it means whether to use objects that has no relationship?
        self.use_orphaned_objects = use_orphaned_objects
        # whether to use object crops
        self.use_object_crops = use_object_crops
        # maximum of objects that appears in image?
        self.max_objects = max_objects
        # If set, we only use a subset of the entire dataset for training
        self.max_samples = max_samples
        # whether to include relationships in training? ablation study?
        self.include_relationships = include_relationships
        # create transform as a list with length of 2
        # containing Resize instance and
        # class that can Convert a PIL Image or numpy.ndarray to tensor
        self.normalize_method = normalize_method
        transform = [T.Resize(image_size), T.ToTensor(), imagenet_preprocess(self.normalize_method)]
        # Compose the transforms
        self.transform = T.Compose(transform)

        # load the values in h5py files into the self.data dictionary
        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                # if got image_path, save it as a list
                if k == 'image_paths':
                    self.image_paths = list(v)
                # otherwise save them as IntTensor
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))
        # find the suitable collate function to make up batches
        self.retrieve_sampling = None
        if self.use_object_crops:
            self.mem_bank_path = kwargs["mem_bank_path"]
            self.crop_file_csv = kwargs["crop_file_csv"]
            self.crop_file_pickle = kwargs["crop_file_pickle"]
            self.top10_crop_ids = kwargs["top10_crop_ids"]
            self.crop_size = kwargs.get("crop_size", None)
            self.build_canvas = kwargs.get("build_canvas", False)
            self.retrieve_sampling = kwargs.get("retrieve_sampling", "random")
            self.crop_num = kwargs.get("crop_num", 1)
            self.load_membank()
        # We group the item into three subgroups:
        # [img-indexed], [obj-indexed], [triple-indexed]. Then we unify the
        # all the collate function into one
        self.collate_fn = vg_collate_fn

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - selected_crops: Object crops selected by Selector, FloatTensor of shape (O, C, H/2, W/2)
        - original_crops: Object crops in ground-truth images, FloatTensor of shape (O, C, H/2, W/2)
        - canvas_sel: FloatTensor of shape (C, H, W), building with selected_crops
        - canvas_ori: FloatTensor of shape (C, H, W), building with original_crops
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        img_path = os.path.join(self.image_dir, self.image_paths[index])

        # 1. load image
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                # get the size of the original image
                WW, HH = image.size
                # convert to RGB channel, then perform the transformations of the image
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size

        # 2. Prepare the objects
        # initialize obj_idxs_with_rels as an empty set
        obj_items = dict()
        obj_idxs_with_rels = set()
        # initialize obj_idxs_without_rels as full indexes of objects in image
        obj_idxs_without_rels = set(
            range(self.data['objects_per_image'][index]))
        # loop over the relationships
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s_category = self.data['object_names'][index, s].item()
            o_category = self.data['object_names'][index, o].item()
            p_category = self.data['relationship_predicates'][index, r_idx].item()
            # Store the connected objects and their relations, For future retrival of object crops
            if s not in obj_idxs_with_rels:
                obj_items[s] = object_item(
                    s_category, num_objects=self.num_objects,
                    boxes=self.data['object_boxes'][index, s],
                    object_id=self.data['object_ids'][index, s])
            if o not in obj_idxs_with_rels:
                obj_items[o] = object_item(
                    o_category, num_objects=self.num_objects,
                    boxes=self.data['object_boxes'][index, o],
                    object_id=self.data['object_ids'][index, o])

            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            # discard objects with relationships from obj_idxs_without_rels
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        for o_idx in obj_idxs_without_rels:
            obj_items[o_idx] = object_item(
                self.data['object_names'][index, o_idx],
                num_objects=self.num_objects,
                boxes=self.data['object_boxes'][index, o_idx],
                object_id=self.data['object_ids'][index, o_idx]
            )

        # convert to list
        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        # If there are more objects than the max_objects, then sample them
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects - 1)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)

        O = len(obj_idxs) + 1
        # objects: (O, )
        objs = torch.LongTensor(O).fill_(-1)
        # object boxes: (O, 4)
        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}
        selected_crops = []
        original_crops = []
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx]
            x, y, w, h = self.data['object_boxes'][index, obj_idx]
            # normalize the object coordinates
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i
            if self.use_object_crops:
                sel_crops = []
                ori_crops = []
                for _ in range(self.crop_num):
                    _crops = self.retrieve_object(obj_items[obj_idx])
                    sel_crops.append(_crops[0])
                    ori_crops.append(_crops[1])
                selected_crops.append(torch.cat(sel_crops, dim=0))
                original_crops.append(torch.cat(ori_crops, dim=0))

        # The last object will be the special __image__ object
        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']


        triples = []
        for r_idx in range(self.data['relationships_per_image'][index]):
            if not self.include_relationships:
                break
            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        # garentee there exists at least one relation to avoid backward errors
        if len(triples) == 0:
            # print("No triples found. Re-Sample.")
            return self[np.random.randint(len(self))]

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])
            if len(triples) == 0:
                print("Cannot find one relations.")

        triples = torch.LongTensor(triples)
        img_tuple = (image,)
        obj_tuple = (objs, boxes,)
        triple_tuple = (triples, )
        if self.use_object_crops:
            # add an noise feature map for __image__ #
            selected_crops.append(torch.zeros_like(selected_crops[0]))
            selected_crops = torch.stack(selected_crops, dim=0)
            obj_tuple += (selected_crops,)
            original_crops.append(torch.zeros_like(original_crops[0]))
            original_crops = torch.stack(original_crops, dim=0)
            obj_tuple += (original_crops,)

            if self.build_canvas:
                assert self.crop_num <= 1, "Please disable [build canvas] in Model Options"
                canvas_sel = make_canvas_baseline(boxes, selected_crops, H, W)
                img_tuple += (canvas_sel, )
                canvas_ori = make_canvas_baseline(boxes, original_crops, H, W)
                img_tuple += (canvas_ori, )
        return img_tuple, obj_tuple, triple_tuple

    def retrieve_object(self, object_item, image_id=None):
        """
        Retrieve the object crops from the memory bank
        input:
            object_item: [object category, connected subjects,
                          connected objects, normalized_size]
            image_id: To exclude the specified image
        output: retrieved object crops (C, HH, WW)
        """
        return self.VgoDataset.retrieve(object_item, image_id)


    def load_membank(self):
        """
        Load memory bank
        Create VG Objects Dataset
        and prepare for [retrieve_object]
        """
        self.VgoDataset = Cropped_VG_Dataset(
                self.crop_file_csv,
                self.crop_file_pickle,
                self.mem_bank_path,
                self.top10_crop_ids,
                output_size=self.crop_size,
                retrieve_sampling=self.retrieve_sampling,
                normalize_method=self.normalize_method,
            )
        print("Creating Visual Genome Objects Dataset Done.")

        return 0


def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:
    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
      triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
      triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    TODO: collate the batch to make it suitable for nn.DataParallel
    """

    # batch is a list, and each element is (image, objs, boxes, triples)
    num_gpu = torch.cuda.device_count()
    if len(batch) % num_gpu:
        raise ValueError("Batch size should be divided by num_gpu (batch size: {}).".format(len(batch)))

    all_imgs = [ [] for _ in batch[0][0]]
    all_objs = [ [] for _ in batch[0][1]]
    all_triples =  [ [] for _ in batch[0][2]]
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, triples) in enumerate(batch):
        for j in range(len(img)):
            all_imgs[j].append(img[j][None])
        O, T = objs[0].size(0), triples[0].size(0)
        for j in range(len(objs)):
            all_objs[j].append(objs[j])
        for j in range(len(triples)):
            if j == 0:
                # here we assume the first element
                # is [subject, predicate, object] triples
                temp_triples = triples[0].clone()
                temp_triples[:, 0] += obj_offset
                temp_triples[:, 2] += obj_offset
                all_triples[j].append(temp_triples)
            else:
                all_triples[j].append(triples[j])
        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    # get the chunk size for multigpu
    scatter_size_obj, scatter_size_triple = [], []
    counter_obj = 0
    counter_triple = 0
    subbatch_size = len(batch) // num_gpu
    for i in range(len(batch)):
        counter_obj += all_objs[0][i].size(0)
        counter_triple += all_triples[0][i].size(0)
        if (i+1) % subbatch_size == 0:
            scatter_size_obj.append(counter_obj)
            scatter_size_triple.append(counter_triple)
            counter_obj = 0
            counter_triple = 0

    # torch concatenate (convert to tensor?)
    all_imgs = tuple([torch.cat(v) for v in all_imgs])
    all_objs = tuple([torch.cat(v) for v in all_objs])
    all_triples = tuple([torch.cat(v) for v in all_triples])
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out, scatter_size_obj, scatter_size_triple


def get_scatter_size(minibatches):
    scatter_size = [len(b) for b in minibatches]
    # flag_tensor = torch.zeros(sum(batch_lens)).long()
    # start_id = 0
    # for i, batch in enumerate(minibatches):
    #     flag_tensor[start_id:(start_id + batch.size(0)] = i
    return scatter_size


def vg_uncollate_fn(batch):
    """
    Inverse operation to the above.
    """
    imgs, objs, triples, obj_to_img, triple_to_img = batch
    out = []
    obj_offset = -1
    for i in range(imgs[0].size(-1)):
        cur_img = tuple([v for v in imgs[i]])
        o_idxs = (obj_to_img == i).nonzero().view(-2)
        t_idxs = (triple_to_img == i).nonzero().view(-2)
        cur_objs = tuple([v[o_idxs] for v in objs])
        cur_triples = []
        for j in range(len(triples)):
            if j == 0:
                t_triples = triples[0][t_idxs].clone()
                t_triples[:, -1] -= obj_offset
                t_triples[:, 1] -= obj_offset
                cur_triples.append(t_triples)
            else:
                cur_triples.append(triples[j][t_idxs])
        cur_triples = tuple(cur_triples)
        obj_offset += cur_objs[0].size(-1)
        out.append((cur_img, cur_objs, cur_triples))
    return out


def vg_uncollate_fn_with_object_crop(batch):
    """
    Inverse operation to the above.
    """
    imgs, objs, boxes, triples, obj_to_img, triple_to_img, object_crops = batch
    out = []
    obj_offset = -1
    for i in range(imgs.size(-1)):
        cur_img = imgs[i]
        o_idxs = (obj_to_img == i).nonzero().view(-2)
        t_idxs = (triple_to_img == i).nonzero().view(-2)
        cur_objs = objs[o_idxs]
        cur_boxes = boxes[o_idxs]
        cur_triples = triples[t_idxs].clone()
        cur_triples[:, -1] -= obj_offset
        cur_triples[:, 1] -= obj_offset
        cur_object_crops = object_crops[o_idxs]
        obj_offset += cur_objs.size(-1)
        out.append((cur_img, cur_objs, cur_boxes,
                    cur_triples, cur_object_crops))
    return out
