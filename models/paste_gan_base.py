import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.box_utils as box_utils
from .graph import GraphTripleConv, GraphTripleConvNet
from .crn import RefinementNetwork
from utils.layout import boxes_to_layout, masks_to_layout
from .layers import build_mlp
from models.utils.batchnorm import BatchNorm2d
from utils.canvas import make_canvas_baseline


class PasteGAN_Base(nn.Module):
    def __init__(self, vocab, image_size=(64, 64),):
        super(PasteGAN_Base, self).__init__()
        # We used to have some additional arguments:
        # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
        self.vocab = vocab
        self.image_size = image_size
        self.pred_embeddings = None
        self.gconv = None
        self.gconv_net = None
        self.box_net = None
        self.mask_net = None
        self.rel_aux_net = None
        self.refinement_net = None

    def _build_mask_net(self, num_objs, dim, mask_size,
            output_dim=None, start_size=1):
        if output_dim is None:
            output_dim = num_objs
        layers, cur_size = [], start_size
        if cur_size >= mask_size:
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        while cur_size < mask_size:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            cur_size *= 2
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, objs, triples, obj_to_img=None, triple_to_img=None,
                boxes_gt=None, selected_crops=None, original_crops=None,
                **kwargs):
        raise NotImplementedError

    def encode_scene_graphs(self, scene_graphs):
        """
        Encode one or more scene graphs using this model's vocabulary. Inputs to
        this method are scene graphs represented as dictionaries like the following:

        {
          "objects": ["cat", "dog", "sky"],
          "relationships": [
            [0, "next to", 1],
            [0, "beneath", 2],
            [2, "above", 1],
          ]
        }

        This scene graph has three relationshps: cat next to dog, cat beneath sky,
        and sky above dog.

        Inputs:
        - scene_graphs: A dictionary giving a single scene graph, or a list of
          dictionaries giving a sequence of scene graphs.

        Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
        same semantics as self.forward. The returned LongTensors will be on the
        same device as the model parameters.
        """
        if isinstance(scene_graphs, dict):
            # We just got a single scene graph, so promote it to a list
            scene_graphs = [scene_graphs]

        objs, triples, obj_to_img, boxes = [], [], [], []
        obj_offset = 0
        for i, sg in enumerate(scene_graphs):
            # Insert dummy __image__ object and __in_image__ relationships
            sg['objects'].append('__image__')
            sg['boxes'].append([0.0, 0.0, 1.0, 1.0])
            image_idx = len(sg['objects']) - 1
            for j in range(image_idx):
                sg['relationships'].append([j, '__in_image__', image_idx])

            for obj in sg['objects']:
                obj_idx = self.vocab['object_name_to_idx'].get(obj, None)
                if obj_idx is None:
                    raise ValueError('Object "%s" not in vocab' % obj)
                objs.append(obj_idx)
                obj_to_img.append(i)
            for s, p, o in sg['relationships']:
                pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
                if pred_idx is None:
                    raise ValueError('Relationship "%s" not in vocab' % p)
                triples.append([s + obj_offset, pred_idx, o + obj_offset])
            obj_offset += len(sg['objects'])
            # added for returning the boxes
            for box in sg['boxes']:
                boxes.append(box)
            ###
        device = next(self.parameters()).device
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
        # added for converting the list to torch tensor
        boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
        return objs, triples, obj_to_img, boxes

    def forward_json(self, scene_graphs, object_crops=None):
        """
        Convenience method that combines encode_scene_graphs, object_crops
        and forward.
        """
        objs, triples, obj_to_img, boxes = self.encode_scene_graphs(
            scene_graphs)

        device = next(self.parameters()).device
        if object_crops is not None:
            return self.forward(objs, triples, obj_to_img, triple_to_img=None,
                                boxes_gt=boxes, selected_crops=object_crops,
                                original_crops=object_crops)
        else:
            return self.forward(objs, triples, obj_to_img, boxes_gt=boxes)
