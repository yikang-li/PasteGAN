import functools
import os
import json
import math
from collections import defaultdict
import random
import time
import pyprind
import glog as log
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from imageio import imwrite
from scipy.misc import imresize

from datasets import imagenet_deprocess_batch
import datasets
import models
import models.perceptual
from utils.losses import get_gan_losses
from utils import timeit, LossManager
from options.opts import args, options
from utils.logger import Logger
from utils import tensor2im
from utils.training_utils import add_loss, check_model, calculate_model_losses
from models.utils import DiscriminatorDataParallel, GeneratorDataParallel
from utils.training_utils import visualize_sample, unpack_batch
import utils.visualization as vis

from scripts.compute_inception_score import get_inception_score
from scripts.compute_diversity_score import compute_diversity_score

torch.backends.cudnn.benchmark = True


def main():
    global args, options
    print(args)

    device = torch.device('cuda')
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    log.info("Building loader...")
    vocab, train_loader, val_loader = datasets.build_loaders(options["data"])
    normalize_method = options["data"]["data_opts"].get('normalize_method', 'imagenet')

    log.info("Building Generative Model...")
    model, model_kwargs = models.build_model(
        options["generator"],
        vocab,
        image_size=options["data"]["image_size"],
        checkpoint_start_from=args.checkpoint_start_from)
    model.type(float_dtype)
    print(model)
    model.eval()
    model.to(device)

    # create folder for output images
    if not os.path.exists('./output/result_image_1'):
        os.mkdir('./output/result_image_1')
    if not os.path.exists('./output/result_image_2'):
        os.mkdir('./output/result_image_2')

    images = []
    for epoch in range(2):
        images_per = []
        for iter, batch in enumerate(pyprind.prog_bar(val_loader,
                                      title="[Epoch {}/{}]".format(epoch, 1),
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
            with torch.no_grad():
                model_out = model(objs, triples, obj_to_img, triple_to_img,
                                  boxes_gt=model_boxes,
                                  canvases_sel=canvases_sel,
                                  canvases_ori=canvases_ori,
                                  selected_crops=selected_crops,
                                  original_crops=original_crops)
            imgs_pred, imgs_rcst, boxes_pred, others = model_out

            if epoch == 0:
                img = imagenet_deprocess_batch(imgs_pred, normalize_method=normalize_method)
                if selected_crops is not None and args.sv_crops != 0:
                    crops = imagenet_deprocess_batch(selected_crops)
                    path_i = os.path.join('./output/result_image_1/%d' % iter)
                else:
                    path_i = os.path.join('./output/result_image_1')
                for i in range(img.shape[0]):
                    img_np = img[i].numpy().transpose(1, 2, 0)
                    if not os.path.exists(path_i):
                        os.mkdir(path_i)
                    img_path = os.path.join(path_i, 'img_%d.png' % iter)
                    imwrite(img_path, img_np)
                    if selected_crops is not None and args.sv_crops != 0:
                        for j in range(crops.shape[0]):
                            crop = crops[j].numpy().transpose(1, 2, 0)
                            crop_path = os.path.join(path_i, 'crop_%d.png' % j)
                            imwrite(crop_path, crop)
                    images_per.append(img_np)
            else:
                img = imagenet_deprocess_batch(imgs_rcst, normalize_method=normalize_method)
                if original_crops is not None:
                    crops = imagenet_deprocess_batch(original_crops)
                    path_i = os.path.join('./output/result_image_2/%d' % iter)
                else:
                    path_i = os.path.join('./output/result_image_2')
                for i in range(img.shape[0]):
                    img_np = img[i].numpy().transpose(1, 2, 0)
                    if not os.path.exists(path_i):
                        os.mkdir(path_i)
                    img_path = os.path.join(path_i, 'img_%d.png' % iter)
                    imwrite(img_path, img_np)
                    if original_crops is not None and args.sv_crops != 0:
                        for j in range(crops.shape[0]):
                            crop = crops[j].numpy().transpose(1, 2, 0)
                            crop_path = os.path.join(path_i, 'crop_%d.png' % j)
                            imwrite(crop_path, crop)
                    images_per.append(img_np)

        images.append(images_per)

    log.info("Start to compute inception score...")
    IS_mean, IS_std = get_inception_score(images[0])
    print('1st Inception Score mean: ', IS_mean)
    print('1st Inception Score std: ', IS_std)
    IS_mean, IS_std = get_inception_score(images[1])
    print('2nd Inception Score mean: ', IS_mean)
    print('2nd Inception Score std: ', IS_std)
    log.info("Start to compute diversity score...")
    DS_mean, DS_std = compute_diversity_score(images[0], images[1], use_gpu=True)
    print('Diveristy Score mean: ', DS_mean)
    print('Diversity Score std: ', DS_std)


if __name__ == '__main__':
    main()
