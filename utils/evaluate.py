import pyprind
import numpy as np
import torch
import glog as log

from datasets import imagenet_deprocess_batch
from utils.training_utils import unpack_batch
from scripts.compute_inception_score import get_inception_score


def evaluate(model, data_loader, options):
    normalize_method = options["data"]["data_opts"].get('normalize_method', 'imagenet')
    model.eval()
    log.info("Evaluating with Inception Scores.")
    images = []
    for iter, batch in enumerate(pyprind.prog_bar(data_loader,
                                  title="[Generating Images]",
                                  width=50)):
        ######### unpack the data #########
        batch = unpack_batch(batch, options)
        (imgs, canvases_sel, canvases_ori,
            objs, boxes, selected_crops,
            original_crops, triples, predicates,
            obj_to_img, triple_to_img,
            scatter_size_obj, scatter_size_triple) = batch
        ###################################
        model_boxes = boxes
        model_masks = masks
        with torch.no_grad():
            model_out = model(objs, triples, obj_to_img, triple_to_img,
                              boxes_gt=model_boxes,
                              selected_crops=selected_crops,
                              original_crops=original_crops,
                              scatter_size_obj=scatter_size_obj,
                              scatter_size_triple=scatter_size_triple,)
        imgs_pred, imgs_rcst, boxes_pred, others = model_out
        img = imagenet_deprocess_batch(imgs_pred, normalize_method=normalize_method)
        for i in range(img.shape[0]):
            img_np = img[i].numpy().transpose(1, 2, 0)
            images.append(img_np)
    log.info("Computing inception scores...")
    mean, std = get_inception_score(images)
    return mean, std


if __name__ == '__main__':
    evaluate()
