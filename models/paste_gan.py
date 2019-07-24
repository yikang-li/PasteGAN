import math
import copy
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
import glog as log

import utils.box_utils as box_utils
from .graph import GraphTripleConv, GraphTripleConvNet
from .graph2d import GraphTripleConv2d, GraphTripleConv2dNet
from .crn import RefinementNetwork
from utils.bilinear import uncrop_bbox
from utils.layout import boxes_to_layout, masks_to_layout, boxes_to_layouts
from .layers import build_mlp, ResidualBlock
from models.utils.batchnorm import BatchNorm2d
from .paste_gan_base import PasteGAN_Base
from .layers import GlobalAvgPool, build_cnn, get_activation, get_normalization_2d
from utils import set_trainable
from . import crop_encoder
from .crop_encoder import CropEncoder


class PasteGAN(PasteGAN_Base):
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 gconv_valid_edge_only=False,
                 refinement_dims=(1024, 512, 256, 128, 64),
                 normalization='batch', activation='leakyrelu-0.2',
                 mlp_normalization='none', canvas_noise_dim=0,
                 crop_encoder=None, generator_kwargs=None,
                 transform_residual=False, gconv2d_num_layers=4,
                 crop_matching_loss=False, class_related_bbox=False,
                 use_flow_net=False, use_mask_net=False,
                 mask_size=None, use_canvas_res=True,
                 **kwargs):

        super(PasteGAN, self).__init__(vocab, image_size, )
        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)
        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)
        self.crop_embedding_dim = crop_encoder["crop_embedding_dim"]
        self.canvas_noise_dim = canvas_noise_dim
        self.use_canvas_res = use_canvas_res
        self.class_related_bbox = class_related_bbox
        self.crop_matching_loss = crop_matching_loss
        self.gconv2d_num_layers = gconv2d_num_layers
        self.crop_encoder = CropEncoder(
            output_D=self.crop_embedding_dim,
            num_categories=num_objs if crop_encoder["category_aware_encoder"] else 1,
            cropEncoderArgs=crop_encoder["crop_encoder_kwargs"],
            decoder_dims = crop_encoder.get('decoder_dims', None),
            pooling=crop_encoder["pooling"],
        )
        assert gconv_num_layers > 2
        gconv_kwargs = {
            'input_dim': embedding_dim,
            'output_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'mlp_normalization': mlp_normalization,
        }
        self.gconv = GraphTripleConv(**gconv_kwargs)
        gconv_kwargs = {
            'input_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers - 2,
            'mlp_normalization': mlp_normalization,
        }
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)
        self.obj_vec_transform = nn.Sequential(
            nn.Linear(gconv_dim, self.crop_embedding_dim),
            nn.ReLU(),
        )
        if self.gconv2d_num_layers > 0:
            gconv2d_kwargs = {
                'input_dim': self.crop_embedding_dim,
                'input_dim_pred': gconv_dim,
                'hidden_dims': [gconv_hidden_dim] * gconv2d_num_layers,
                'output_dim': self.crop_embedding_dim,
                'normalization': normalization,
                'activation': activation,
                'transform_residual': transform_residual,
                'use_flow_net': use_flow_net,
                'use_mask_net': use_mask_net,
                'valid_edge_only': gconv_valid_edge_only,
                'vocab': self.vocab,
            }
            self.object_refiner = GraphTripleConv2dNet(**gconv2d_kwargs)
            gconv2d_kwargs = {
                'input_dim': gconv_dim * 2,
                'input_dim_pred': gconv_dim * 2,
                'hidden_dim': gconv_dim * 2,
                'output_dim': gconv_dim * 2,
                'normalization': normalization,
                'activation': activation,
                'last_layer': True,
                'vocab': self.vocab,
                'pooling': 'sum',
                'use_mask_net': False,
            }
            self.obj_img_fuser = GraphTripleConv2d(**gconv2d_kwargs)


        if self.class_related_bbox:
            box_net_dim = 4 * num_objs
        else:
            box_net_dim = 4
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

        crn_input_dims = self.crop_embedding_dim * 2 + canvas_noise_dim
        refinement_kwargs = {
            'dims': (crn_input_dims,) + tuple(refinement_dims),
            'normalization': normalization,
            'activation': 'leakyrelu-0.1',
        }
        self.img_decoder = RefinementNetwork(**refinement_kwargs)

        if self.crop_matching_loss:
            self.crop_encoder_copy = copy.deepcopy(self.crop_encoder)

        if self.use_canvas_res:
            self.canvas_res1 = ResidualBlock(crn_input_dims, normalization='batch')
            self.canvas_res2 = ResidualBlock(crn_input_dims, normalization='batch')
            self.canvas_res3 = ResidualBlock(crn_input_dims, normalization='batch')
            self.canvas_res4 = ResidualBlock(crn_input_dims, normalization='batch')
            self.canvas_res5 = ResidualBlock(crn_input_dims, normalization='batch')
            self.canvas_res6 = ResidualBlock(crn_input_dims, normalization='batch')


    def forward(self, objs, triples, obj_to_img=None, triple_to_img=None,
                boxes_gt=None, selected_crops=None, original_crops=None,
                **kwargs):
        """
        Required Inputs:
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])
        - selected_crops: LongTensor of shape (O, 3, H/2, W/2), giving the selected
          object crops as the source materials of generation.
        - original_crops: LongTensor of shape (O, 3, H/2, W/2), giving the original
          object crops as the source materials of generation.
          (If you don't want to specify the object crops in inference, the Selector
           will select the most-matching crops for the generation.)

        Optional Inputs:
        - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        """
        O, T = objs.size(0), triples.size(0)
        HH_o, WW_o = selected_crops.size(2), selected_crops.size(3)
        s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o, p], dim=1)          # Shape is (T, 2)
        others = {}

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        obj_vecs_orig = obj_vecs
        pred_vecs = self.pred_embeddings(p)
        pred_vecs_orig = pred_vecs

        # GCN to process the input scene graph
        obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        # Box Regressor predicts the bounding box for each object
        boxes_pred = self.box_net(obj_vecs)
        if self.class_related_bbox:
            obj_cat = objs.view(-1, 1) * 4 + torch.arange(end=4, device=objs.device).view(1, -1)
            boxes_pred = boxes_pred.gather(1, obj_cat)
        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt

        # crop encoder encodes the obejct crops
        sel_crops_feat = self.crop_encoder(selected_crops, objs)
        ori_crops_feat = self.crop_encoder(original_crops, objs)

        crop_vec_samples = sel_crops_feat

        H, W = self.image_size
        resized_size = int(H/4)

        if self.gconv2d_num_layers > 0:
            # refine object crops with Crop Refiner
            sel_crops_feat, sel_pred_maps = self.object_refiner(sel_crops_feat, pred_vecs_orig, edges, obj_to_img)
            ori_crops_feat, ori_pred_maps = self.object_refiner(ori_crops_feat, pred_vecs_orig, edges, obj_to_img)

            # concatenate obj_vec (category) & crops (appearance)
            # to get the integral representation of object
            sel_obj_integral = torch.cat([obj_vecs.view(O,-1,1,1).expand_as(sel_crops_feat), sel_crops_feat], dim=1)
            ori_obj_integral = torch.cat([obj_vecs.view(O,-1,1,1).expand_as(ori_crops_feat), ori_crops_feat], dim=1)

            obj_mask = (objs != 0)
            edge_mask = (p == self.vocab['pred_name_to_idx']['__in_image__'])

            # refill the region of bounding boxes to get the original layouts
            sel_obj_integral_layouts = boxes_to_layouts(sel_obj_integral, layout_boxes, H, W)
            ori_obj_integral_layouts = boxes_to_layouts(ori_obj_integral, layout_boxes, H, W)

            sel_pred_maps = torch.cat([pred_vecs.view(T,-1,1,1).expand_as(sel_pred_maps), sel_pred_maps], dim=1)
            ori_pred_maps = torch.cat([pred_vecs.view(T,-1,1,1).expand_as(ori_pred_maps), ori_pred_maps], dim=1)
            sel_pred_maps = boxes_to_layouts(sel_pred_maps[edge_mask], layout_boxes[obj_mask], H, W)
            ori_pred_maps = boxes_to_layouts(ori_pred_maps[edge_mask], layout_boxes[obj_mask], H, W)

            # Object-Image Fuser fuse all the original layouts into one scene canvas
            scene_canvas_pred, _ = self.obj_img_fuser(
                sel_obj_integral_layouts,
                sel_pred_maps,
                edges[edge_mask],
                obj_to_img,)

            scene_canvas_rcst, _ = self.obj_img_fuser(
                ori_obj_integral_layouts,
                ori_pred_maps,
                edges[edge_mask],
                obj_to_img,)

        # add noise to image generation
        if self.canvas_noise_dim > 0:
            N, C, H, W = layout.size()
            noise_shape = (N, self.canvas_noise_dim, H, W)
            canvas_pred_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                       device=layout.device)
            canvas_rcst_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                       device=layout.device)
            scene_canvas_pred = torch.cat([scene_canvas_pred, canvas_pred_noise], dim=1)
            scene_canvas_rcst = torch.cat([scene_canvas_rcst, canvas_rcst_noise], dim=1)

        # use resblock for better refining
        if self.use_canvas_res:
            scene_canvas_pred = self.canvas_res1(scene_canvas_pred)
            scene_canvas_pred = self.canvas_res2(scene_canvas_pred)
            scene_canvas_pred = self.canvas_res3(scene_canvas_pred)
            scene_canvas_pred = self.canvas_res4(scene_canvas_pred)
            scene_canvas_pred = self.canvas_res5(scene_canvas_pred)
            scene_canvas_pred = self.canvas_res6(scene_canvas_pred)

            scene_canvas_rcst = self.canvas_res1(scene_canvas_rcst)
            scene_canvas_rcst = self.canvas_res2(scene_canvas_rcst)
            scene_canvas_rcst = self.canvas_res3(scene_canvas_rcst)
            scene_canvas_rcst = self.canvas_res4(scene_canvas_rcst)
            scene_canvas_rcst = self.canvas_res5(scene_canvas_rcst)
            scene_canvas_rcst = self.canvas_res6(scene_canvas_rcst)

        # Image Decoder (CRN) decodes the scene canvas into image
        img_pred = self.img_decoder(scene_canvas_pred)
        img_rcst = self.img_decoder(scene_canvas_rcst)

        # calculate object feature matching Loss
        if self.crop_matching_loss:
            from utils.bilinear import crop_bbox_batch
            from utils import set_trainable_param
            import copy
            valid_obj_mask = objs != self.vocab['object_name_to_idx']['__image__']
            generated_crops = crop_bbox_batch(img_pred,
                    layout_boxes[valid_obj_mask],
                    obj_to_img[valid_obj_mask],
                    HH_o, WW_o)
            self.crop_encoder_copy.load_state_dict(self.crop_encoder.state_dict())
            generated_crops = self.crop_encoder(generated_crops, objs[valid_obj_mask])
            if isinstance(generated_crops, tuple):
                # we want the [mean] to be similar to the sampled_value
                generated_crops = generated_crops[3]
            crop_matching_loss = F.l1_loss(generated_crops, crop_vec_samples[valid_obj_mask])
            others["CML"] = crop_matching_loss

        return img_pred, img_rcst, boxes_pred, others
