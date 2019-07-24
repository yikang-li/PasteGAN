import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel as DataParallel_raw
import numpy as np


def rescatter_obj_and_triples(obj_to_img, triple_to_img, triples, scatter_size_obj, device_ids):
    num_images = obj_to_img.max().item() + 1
    batch_size = num_images // len(device_ids)
    obj_to_img = torch.fmod(obj_to_img, batch_size).long() if obj_to_img is not None else None
    triple_to_img = torch.fmod(triple_to_img, batch_size).long() if triple_to_img is not None else None
    cumsum_obj = np.cumsum([0, ] + scatter_size_obj)
    mappings = list(range(len(obj_to_img)))
    thres_id = 1
    for i in range(len(mappings)):
        while mappings[i] >= cumsum_obj[thres_id]:
            thres_id += 1
        mappings[i] = mappings[i] - cumsum_obj[thres_id-1]
    mappings = torch.cuda.LongTensor(mappings).type_as(triples)
    new_triples = torch.zeros_like(triples).copy_(triples)
    new_triples[:, 0] = mappings[new_triples[:, 0]]
    new_triples[:, 2] = mappings[new_triples[:, 2]]
    return obj_to_img, triple_to_img, new_triples

class GeneratorDataParallel(DataParallel_raw):
    """
    we do the scatter outside of the DataPrallel.
    input: Scattered Inputs without kwargs.
    """

    def __init__(self, module):
        # Disable all the other parameters
        super(GeneratorDataParallel, self).__init__(module)


    def forward(self, objs, triples, obj_to_img, triple_to_img=None,
                boxes_gt=None,
                imgs=None,
                selected_crops=None,
                original_crops=None,
                scatter_size_obj=None, scatter_size_triple=None):

        if scatter_size_obj is None or scatter_size_triple is None or \
                len(self.device_ids) == 1:
            inputs = (objs, triples, )
            inputs_kwargs = {
                    "obj_to_img": obj_to_img,
                    "triple_to_img": triple_to_img,
                    "boxes_gt": boxes_gt,
                    "imgs": imgs,
                    "selected_crops": selected_crops,
                    "original_crops": original_crops,
            }
            return self.module(*inputs, **inputs_kwargs)
        if not self.device_ids:
            raise NotImplementedError
        # prepare the obj_to_img and triples
        obj_to_img, triple_to_img, triples = rescatter_obj_and_triples(
                obj_to_img, triple_to_img, triples,
                scatter_size_obj, self.device_ids)

        nones = [None for _ in scatter_size_obj]
        objs = torch.cuda.comm.scatter(objs, self.device_ids, chunk_sizes=scatter_size_obj)
        triples = torch.cuda.comm.scatter(triples, self.device_ids, chunk_sizes=scatter_size_triple)
        boxes_gt = torch.cuda.comm.scatter(boxes_gt, self.device_ids, chunk_sizes=scatter_size_obj) if boxes_gt is not None else nones
        obj_to_img = torch.cuda.comm.scatter(obj_to_img, self.device_ids, chunk_sizes=scatter_size_obj) if obj_to_img is not None else nones
        triple_to_img = torch.cuda.comm.scatter(triple_to_img, self.device_ids, chunk_sizes=scatter_size_triple) if triple_to_img is not None else nones
        imgs = torch.cuda.comm.scatter(imgs, self.device_ids) if imgs is not None else nones
        selected_crops = torch.cuda.comm.scatter(selected_crops, self.device_ids, chunk_sizes=scatter_size_obj) if selected_crops is not None else nones
        original_crops = torch.cuda.comm.scatter(original_crops, self.device_ids, chunk_sizes=scatter_size_obj) if original_crops is not None else nones
        inputs = (objs, triples, )
        inputs = list(zip(*inputs))
        inputs_kwargs = [
                {
                    "obj_to_img": v[0],
                    "triple_to_img": v[1],
                    "boxes_gt": v[2],
                    "selected_crops": v[3],
                    "original_crops": v[4],
                    "imgs": v[5],
                } for v in zip(obj_to_img,
                               triple_to_img,
                               boxes_gt,
                               selected_crops,
                               original_crops,
                               imgs,)
        ]

        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, inputs, inputs_kwargs)
        return self.gather(outputs, self.output_device)



class DiscriminatorDataParallel(DataParallel_raw):
    """
    we do the scatter outside of the DataPrallel.
    input: Scattered Inputs without kwargs.
    """

    def __init__(self, module):
        # Disable all the other parameters
        super(DiscriminatorDataParallel, self).__init__(module)

    def forward(self, imgs, objs, boxes, obj_to_img,
                object_crops=None,
                triple_to_img=None, triples = None,
                scatter_size_obj=None, scatter_size_triple=None,):

        if scatter_size_obj is None or len(self.device_ids) == 1:
            inputs_kwargs = {
                    "triple_to_img": triple_to_img,
                    "triples": triples,
                    "object_crops": object_crops,
            }
            return self.module(imgs, objs, boxes, obj_to_img, **inputs_kwargs)

        if not self.device_ids:
            raise NotImplementedError

        batch_size = len(imgs) // len(self.device_ids)
        if triples is None:
            obj_to_img = torch.fmod(obj_to_img, batch_size).long()
            triple_to_img = torch.fmod(triple_to_img, batch_size).long() if triple_to_img is not None else None
        else:
            # prepare the obj_to_img and triples
            obj_to_img, triple_to_img, triples = rescatter_obj_and_triples(
                    obj_to_img, triple_to_img, triples,
                    scatter_size_obj, self.device_ids)
        # prepare the obj_to_img and triples
        nones = [None for _ in scatter_size_obj]
        imgs = torch.cuda.comm.scatter(imgs, self.device_ids)
        objs = torch.cuda.comm.scatter(objs, self.device_ids, chunk_sizes=scatter_size_obj)
        boxes = torch.cuda.comm.scatter(boxes, self.device_ids, chunk_sizes=scatter_size_obj)
        obj_to_img = torch.cuda.comm.scatter(obj_to_img, self.device_ids, chunk_sizes=scatter_size_obj)
        triple_to_img = torch.cuda.comm.scatter(triple_to_img, self.device_ids, chunk_sizes=scatter_size_triple) if scatter_size_triple is not None else nones
        triples =  torch.cuda.comm.scatter(triples, self.device_ids, chunk_sizes=scatter_size_triple) if scatter_size_triple is not None else nones
        object_crops = torch.cuda.comm.scatter(object_crops, self.device_ids, chunk_sizes=scatter_size_obj) if object_crops is not None else nones
        inputs = list(zip(imgs, objs, boxes, obj_to_img,))
        inputs_kwargs = [
                {
                        "triples": v[0],
                        "triple_to_img": v[1],
                        "object_crops": v[2],
                } for v in zip(triples, triple_to_img, object_crops)
        ]
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, inputs, inputs_kwargs)
        return self.gather(outputs, self.output_device)
