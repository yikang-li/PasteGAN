import os
import json
import math
from collections import defaultdict
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from torchvision.models import inception_v3
from pyprind import prog_bar

from datasets import imagenet_deprocess_batch
import datasets
import models
from utils.metrics import jaccard
from utils import tensor2im
from utils.visualization import draw_scene_graph


def add_loss_with_tensor(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss

def unpack_batch(batch, options):
    canvases_sel, canvases_ori, selected_crops, original_crops = None, None, None, None
    batch, scatter_size_obj, scatter_size_triple = batch
    img_items, obj_items, triple_items, obj_to_img, triple_to_img = batch
    img_items = [data.cuda() for data in img_items]
    obj_items = [data.cuda() for data in obj_items]
    triple_items = [data.cuda() for data in triple_items]
    obj_to_img = obj_to_img.cuda()
    triple_to_img = triple_to_img.cuda()
    if len(img_items) == 3:
        imgs, canvases_sel, canvases_ori = img_items
    else:
        assert False
    if len(obj_items) == 4 and options["data"]["data_opts"]["dataset"] == "visual_genome":
        objs, boxes, selected_crops, original_crops = obj_items
    elif len(obj_items) == 4 and options["data"]["data_opts"]["dataset"] == "coco":
        objs, boxes, selected_crops, original_crops = obj_items
    else:
        assert False
    if len(triple_items) == 1:
        triples, = triple_items
    else:
        assert False
    predicates = triples[:, 1]

    return (imgs, canvases_sel, canvases_ori,
            objs, boxes, selected_crops,
            original_crops, triples, predicates,
            obj_to_img, triple_to_img,
            scatter_size_obj, scatter_size_triple)

def visualize_sample(model, batch, vocab):
    (imgs, canvases_sel, canvases_ori,
        objs, boxes, selected_crops,
        original_crops, triples, predicates,
        obj_to_img, triple_to_img,
        scatter_size_obj, scatter_size_triple) = batch

    samples = []
    # add the ground-truth images
    samples.append(imgs[:1])

    # add the canvases building with original crops
    if canvases_ori is not None:
        samples.append(canvases_ori[:1])

    with torch.no_grad():
        model_out = model(objs, triples, obj_to_img, triple_to_img,
                          boxes_gt=boxes,
                          selected_crops=selected_crops,
                          original_crops=original_crops,
                          scatter_size_obj=scatter_size_obj,
                          scatter_size_triple=scatter_size_triple)

        # add the reconstructed images
        samples.append(model_out[1][:1])

        # add the canvases building with selected crops
        if canvases_sel is not None:
            samples.append(canvases_sel[:1])

        # add the generated images
        samples.append(model_out[0][:1])

        model_out = model(objs, triples, obj_to_img,  triple_to_img,
                          boxes_gt=boxes,
                          selected_crops=selected_crops,
                          original_crops=original_crops,
                          scatter_size_obj=scatter_size_obj,
                          scatter_size_triple=scatter_size_triple)
        # add the generated images
        samples.append(model_out[0][:1])

        model_out = model(objs, triples, obj_to_img, triple_to_img,
                          selected_crops=selected_crops,
                          original_crops=original_crops,
                          scatter_size_obj=scatter_size_obj,
                          scatter_size_triple=scatter_size_triple)
        # add the generated images
        samples.append(model_out[0][:1])
    samples = torch.cat(samples, dim=3)
    samples = {
            "samples": tensor2im(
                imagenet_deprocess_batch(samples, rescale=True).squeeze(0)
            )
    }
    # Draw Scene Graphs
    sg_array = draw_scene_graph(objs[obj_to_img == 0],
                     triples[triple_to_img == 0],
                     vocab=vocab)
    samples["scene_graph"] = sg_array
    return samples


def check_model(args, options, t, loader, model, vocab):
    training_status = model.training
    model.eval()
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    num_samples = 0
    all_losses = defaultdict(list)
    total_iou = 0
    total_boxes = 0
    inception_module = nn.DataParallel(inception_v3(pretrained=True, transform_input=False).cuda())
    inception_module.eval()
    preds = []
    with torch.no_grad():
        # To avoid influence to the running_mean/var of BatchNorm Layers
        for batch in prog_bar(loader, title="[Validating]", width=50):
            ######### unpack the data #########
            batch = unpack_batch(batch, options)
            (imgs, canvases_sel, canvases_ori, objs, boxes,
                selected_crops, original_crops,
                triples, predicates,
                obj_to_img, triple_to_img,
                scatter_size_obj, scatter_size_triple) = batch
            ###################################
            # Run the model as it has been run during training
            model_out = model(objs, triples, obj_to_img, triple_to_img,
                              boxes_gt=boxes,
                              selected_crops=selected_crops,
                              original_crops=original_crops,
                              scatter_size_obj=scatter_size_obj,
                              scatter_size_triple=scatter_size_triple)
            imgs_pred, imgs_rcst, boxes_pred, others = model_out

            skip_pixel_loss = False
            total_loss, losses = calculate_model_losses(
                options["optim"], skip_pixel_loss, imgs, imgs_pred,
                boxes, boxes_pred, get_item=True)

            total_iou += jaccard(boxes_pred, boxes)
            total_boxes += boxes_pred.size(0)

            # check inception scores
            x = F.interpolate(imgs_pred, (299, 299), mode="bilinear")
            x = inception_module(x)
            preds.append(F.softmax(x).cpu().numpy())

            for loss_name, loss_val in losses.items():
                all_losses[loss_name].append(loss_val)
            num_samples += imgs.size(0)

        samples = visualize_sample(model, batch, vocab)
        mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
        avg_iou = total_iou / total_boxes

        # calculate the inception scores
        splits=5
        preds = np.concatenate(preds, axis=0)
        # Now compute the mean kl-div
        split_scores = []
        N = preds.shape[0]
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        inception_score = (np.mean(split_scores), np.std(split_scores))


    batch_data = {
        'objs': objs.detach().cpu().clone(),
        'boxes_gt': boxes.detach().cpu().clone(),
        'triples': triples.detach().cpu().clone(),
        'obj_to_img': obj_to_img.detach().cpu().clone(),
        'triple_to_img': triple_to_img.detach().cpu().clone(),
        'boxes_pred': boxes_pred.detach().cpu().clone(),
    }
    out = [mean_losses, samples, batch_data, avg_iou, inception_score]
    model.train(mode=training_status)
    return tuple(out)


def calculate_model_losses(opts, skip_pixel_loss, img, img_pred,
                           bbox, bbox_pred, get_item=False):
    if get_item:
        add_loss_fn = add_loss
    else:
        add_loss_fn = add_loss_with_tensor
    total_loss = torch.zeros(1).to(img)
    losses = {}

    l1_pixel_weight = opts["l1_pixel_loss_weight"]
    if skip_pixel_loss:
        l1_pixel_weight = 0

    l1_pixel_loss = F.l1_loss(img_pred, img)
    total_loss = add_loss_fn(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                          l1_pixel_weight)

    loss_bbox = F.mse_loss(bbox_pred, bbox)
    total_loss = add_loss_fn(total_loss, loss_bbox, losses, 'bbox_pred',
                          opts["bbox_pred_loss_weight"])

    return total_loss, losses
