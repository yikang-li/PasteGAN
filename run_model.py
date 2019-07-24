import argparse
import json
import os
import PIL
import functools
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
import torchvision.transforms as T

from datasets import imagenet_deprocess_batch
from datasets import imagenet_preprocess
import datasets
import models
from options.opts import args, options
import utils.visualization as vis


torch.backends.cudnn.benchmark = True

def main():

    global args, options
    print(args)
    print(options)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    log.info("Building loader...")
    vocab, train_loader, val_loader = datasets.build_loaders(options["data"])

    log.info("Building Generative Model...")
    model, model_kwargs = models.build_model(
        options["generator"],
        vocab,
        image_size=options["data"]["image_size"],
        checkpoint_start_from=args.checkpoint_start_from)
    model.type(float_dtype)

    # Get the crop size for building transformer
    crop_size = options['data']['data_opts']['crop_size'][0]
    transform = T.Compose([
        T.Resize(crop_size),
        T.ToTensor(),
        imagenet_preprocess(),
    ])

    # Load the scene graphs
    with open(args.scene_graphs_json, 'r') as f:
        scene_graphs = json.load(f)

    if 'crops' in scene_graphs[0].keys():
        # Load the object crops we wanted
        device = torch.device('cuda:0')
        object_crops = []
        for i, sg in enumerate(scene_graphs):
            crop_names = sg['crops']
            for j, name in enumerate(crop_names):
                crops = []
                crop_path = os.path.join(args.samples_path, 'crops', name)
                with open(crop_path, 'rb') as f:
                    with PIL.Image.open(f) as crop:
                        W, H = crop.size
                        crop = transform(crop.convert('RGB'))
                        crops.append(crop)
                        object_crops.append(torch.cat(crops, dim=0))
            object_crops.append(torch.zeros_like(object_crops[0]))
        object_crops = torch.stack(object_crops, dim=0).to(device)
    else:
        object_crops = None

    # Run the model forward
    with torch.no_grad():
        model_out = model.forward_json(scene_graphs, object_crops)
        imgs_pred, imgs_rcst, boxes_pred, others = model_out

    imgs = imagenet_deprocess_batch(imgs_pred)

    if not os.path.exists(args.output_demo_dir):
        os.mkdir(args.output_demo_dir)

    # Save the generated images
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0)
        img_path = os.path.join(args.output_demo_dir, 'img_%d.png' % i)
        imwrite(img_path, img_np)
    print("Saving images finished.")

    # Draw the scene graphs
    if args.draw_scene_graphs == 1:
        for i, sg in enumerate(scene_graphs):
            sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
            sg_img_path = os.path.join(args.output_demo_dir, 'sg%06d.png' % i)
            imwrite(sg_img_path, sg_img)
        print("Saving scene graph finsished.")


if __name__ == '__main__':
    main()
