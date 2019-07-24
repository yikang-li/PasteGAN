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
from utils.evaluate import evaluate

torch.backends.cudnn.benchmark = True


def main():
    global args, options
    print(args)
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
    print(model)

    optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.learning_rate,
            betas=(args.beta1, 0.999),)

    if (options["optim"]["d_loss_weight"] < 0 or options["optim"]["d_obj_weight"] < 0):
        obj_discriminator = None
        d_obj_kwargs = {}
        log.info("Ignoring Object Discriminator.")
    else:
        obj_discriminator, d_obj_kwargs = models.build_obj_discriminator(
            options["discriminator"], vocab
        )
        log.info("Done Building Object Discriminator.")

    if (options["optim"]["d_loss_weight"] < 0 or options["optim"]["d_img_weight"] < 0):
        img_discriminator = None
        d_img_kwargs = {}
        log.info("Ignoring Image Discriminator.")
    else:
        img_discriminator, d_img_kwargs = models.build_img_discriminator(
            options["discriminator"], vocab
        )
        log.info("Done Building Image Discriminator.")

    perceptual_module = None
    if options["optim"].get("perceptual_loss_weight", -1) > 0 or \
            options["optim"].get("obj_perceptual_loss_weight", -1) > 0:
        perceptual_module = getattr(
            models.perceptual,
            options.get("perceptual", {}).get("arch", "VGGLoss"))()

    gan_g_loss, gan_d_loss = get_gan_losses(options["optim"]["gan_loss_type"])

    if obj_discriminator is not None:
        obj_discriminator.type(float_dtype)
        obj_discriminator.train()
        print(obj_discriminator)
        optimizer_d_obj = torch.optim.Adam(
                filter(lambda x: x.requires_grad, obj_discriminator.parameters()),
                lr=args.learning_rate,
                betas=(args.beta1, 0.999),)

    if img_discriminator is not None:
        img_discriminator.type(float_dtype)
        img_discriminator.train()
        print(img_discriminator)
        optimizer_d_img = torch.optim.Adam(
                filter(lambda x: x.requires_grad, img_discriminator.parameters()),
                lr=args.learning_rate,
                betas=(args.beta1, 0.999),)

    restore_path = None
    if args.resume is not None:
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(
            options["logs"]["output_dir"], args.resume, restore_path)
    if restore_path is not None and os.path.isfile(restore_path):
        log.info('Restoring from checkpoint: {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])

        if obj_discriminator is not None:
            obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
            optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

        if img_discriminator is not None:
            img_discriminator.load_state_dict(checkpoint['d_img_state'])
            optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

        t = checkpoint['counters']['t'] + 1
        if 0 <= args.eval_mode_after <= t:
            model.eval()
        else:
            model.train()
        start_epoch = checkpoint['counters']['epoch'] + 1
        log_path = os.path.join(options["logs"]["output_dir"], args.resume,)
        lr = checkpoint.get('learning_rate', args.learning_rate)
        best_inception = checkpoint["counters"].get("best_inception", (0., 0.))
        options = checkpoint.get("options", options)
    else:
        t, start_epoch, best_inception = 0, 0, (0., 0.)
        lr = args.learning_rate
        checkpoint = {
            'args': args.__dict__,
            'options': options,
            'vocab': vocab,
            'model_kwargs': model_kwargs,
            'd_obj_kwargs': d_obj_kwargs,
            'd_img_kwargs': d_img_kwargs,
            'train_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'train_inception': [],
            'lr': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'val_inception': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
                'best_inception': None,
            },
            'model_state': None, 'model_best_state': None, 'optim_state': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
        }

        log_path = os.path.join(
            options["logs"]["output_dir"],
            options["logs"]["name"] + "-" + time.strftime("%Y%m%d-%H%M%S")
        )
    logger = Logger(log_path)
    log.info("Logging to: {}".format(log_path))
    model = GeneratorDataParallel(model)
    obj_discriminator = DiscriminatorDataParallel(obj_discriminator) if obj_discriminator else None
    img_discriminator = nn.DataParallel(img_discriminator.cuda()) if img_discriminator else None
    perceptual_module = nn.DataParallel(perceptual_module.cuda()) if perceptual_module else None

    if args.evaluate:
        assert args.resume is not None
        if args.evaluate_train:
            log.info("Evaluting the training set.")
            train_mean, train_std = evaluate(model, train_loader, options)
            log.info("Inception score: {} ({})".format(train_mean, train_std))
        log.info("Evaluting the testing set.")
        val_mean, val_std = evaluate(model, val_loader, options)
        log.info("Inception score: {} ({})".format(val_mean, val_std))
        return 0


    isBest = True
    pred_crops = None
    others = None
    for epoch in range(start_epoch, args.epochs):
        if epoch >= args.eval_mode_after and model.training:
            log.info('[Epoch {}/{}] switching to eval mode'.format(epoch, args.epochs))
            model.eval()
            if epoch == args.eval_mode_after:
                optimizer = optim.Adam(
                        filter(lambda x: x.requires_grad, model.parameters()),
                        lr=lr,
                        betas=(args.beta1, 0.999),)
        if epoch >= args.disable_l1_loss_after and options["optim"]["l1_pixel_loss_weight"] > 1e-10:
            log.info('[Epoch {}/{}] Disable L1 Loss'.format(epoch, args.epochs))
            options["optim"]["l1_pixel_loss_weight"] = 0
        start_time = time.time()
        for iter, batch in enumerate(pyprind.prog_bar(train_loader,
                                      title="[Epoch {}/{}]".format(epoch, args.epochs),
                                      width=50)):

            if args.timing:
                print("Loading Time: {} ms".format((time.time() - start_time) * 1000))
            t += 1
            ######### unpack the data #########
            batch = unpack_batch(batch, options)
            (imgs, canvases_sel, canvases_ori,
                objs, boxes, selected_crops,
                original_crops, triples, predicates,
                obj_to_img, triple_to_img,
                scatter_size_obj, scatter_size_triple) = batch
            ###################################
            with timeit('forward', args.timing):
                model_boxes = boxes
                model_out = model(objs, triples, obj_to_img, triple_to_img,
                                  boxes_gt=model_boxes,
                                  selected_crops=selected_crops,
                                  original_crops=original_crops,
                                  scatter_size_obj=scatter_size_obj,
                                  scatter_size_triple=scatter_size_triple)
                """
                imgs_pred: images of generated path
                imgs_rcst: images of reconstructed path
                boxes_pred: predicted bounding boxes output by Box Regressor
                """
                imgs_pred, imgs_rcst, boxes_pred, others = model_out

            if (iter+1) % args.visualize_every == 0:
                training_status = model.training
                model.eval()
                samples = visualize_sample(model, batch, vocab)
                model.train(mode=training_status)
                logger.image_summary(samples, t, tag="vis")

            with timeit('G_loss', args.timing):
                # Skip the pixel loss if not using GT boxes
                skip_pixel_loss = (model_boxes is None)

                # calculate L1 loss between imgs and imgs_self
                total_loss, losses = calculate_model_losses(
                    options["optim"], skip_pixel_loss, imgs, imgs_rcst,
                    boxes, boxes_pred,)

                # crop feature matching for object crop samples
                if others and "CML" in others.keys():
                    weight = options["optim"]["CML_weight"]
                    total_loss = add_loss(total_loss, others["CML"].mean(), losses,
                                          "Crop-Matching-Loss", weight)

                if img_discriminator is not None:
                    weight = options["optim"]["d_loss_weight"] * \
                        options["optim"]["d_img_weight"]
                    scores_fake = img_discriminator(imgs_pred)
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                          'g_gan_img_loss', weight)

                    scores_fake_rcst = img_discriminator(imgs_rcst)
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake_rcst), losses,
                                          'g_gan_img_rcst_loss', weight)

                if obj_discriminator is not None:
                    weight = options["optim"]["d_loss_weight"] * \
                        options["optim"]["d_obj_weight"]
                    scores_fake, ac_loss, pred_crops = obj_discriminator(
                        imgs_pred, objs, boxes, obj_to_img,
                        scatter_size_obj=scatter_size_obj)
                    total_loss = add_loss(total_loss, ac_loss.mean(), losses, 'ac_loss',
                                          options["optim"]["ac_loss_weight"])
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                          'g_gan_obj_loss', weight)

                    scores_fake_rcst, ac_loss_rcst, pred_crops_rcst = obj_discriminator(
                        imgs_rcst, objs, boxes, obj_to_img,
                        scatter_size_obj=scatter_size_obj)
                    total_loss = add_loss(total_loss, ac_loss_rcst.mean(), losses, 'ac_loss_rcst',
                                          options["optim"]["ac_loss_weight"])
                    total_loss = add_loss(total_loss, gan_g_loss(scores_fake_rcst), losses,
                                          'g_gan_obj_rcst_loss', weight)

                if options["optim"].get("obj_perceptual_loss_weight", -1) > 0:
                    if pred_crops_rcst is None:
                        pred_crops_rcst = crop_bbox_batch(imgs_rcst,
                            boxes, obj_to_img,
                            options["discriminator"]["object"]["object_size"])
                    obj_mask = objs.nonzero().view(-1)
                    perceptual_loss = perceptual_module(pred_crops_rcst[obj_mask], original_crops[obj_mask])
                    perceptual_loss = perceptual_loss.mean()
                    weight = options["optim"]["obj_perceptual_loss_weight"]
                    total_loss = add_loss(total_loss, perceptual_loss,
                                          losses, "obj_rcst_perceptual_loss",
                                          weight)

                if options["optim"].get("perceptual_loss_weight", -1) > 0:
                    perceptual_loss = perceptual_module(imgs_rcst, imgs)
                    perceptual_loss = perceptual_loss.mean()
                    weight = options["optim"]["perceptual_loss_weight"]
                    total_loss = add_loss(total_loss, perceptual_loss,
                                          losses, "img_rcst_perceptual_loss",
                                          weight)

                if options["optim"].get("double_ploss_weight", -1) > 0:
                    perceptual_loss = perceptual_module(imgs_pred, imgs)
                    perceptual_loss = perceptual_loss.mean()
                    weight = options["optim"]["double_ploss_weight"]
                    total_loss = add_loss(total_loss, perceptual_loss,
                                          losses, "double_perceptual_loss",
                                          weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                log.warn('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                total_loss.backward()
            optimizer.step()

            total_loss_d = None
            ac_loss_real = None
            ac_loss_fake = None
            d_losses = {}

            with timeit('D_loss', args.timing):
                if obj_discriminator is not None:
                    d_obj_losses = LossManager()
                    imgs_pred_fake = imgs_pred.detach()
                    imgs_rcst_fake = imgs_rcst.detach()

                    scores_real, ac_loss_real, _ = obj_discriminator(
                        imgs, objs, boxes, obj_to_img,
                        scatter_size_obj=scatter_size_obj)
                    scores_fake, ac_loss_fake, _ = obj_discriminator(
                        imgs_pred_fake, objs, boxes, obj_to_img,
                        scatter_size_obj=scatter_size_obj)

                    d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
                    d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
                    d_obj_losses.add_loss(ac_loss_real.mean(), 'd_ac_loss_real')
                    # d_obj_losses.add_loss(ac_loss_fake.mean(), 'd_ac_loss_fake')

                    scores_fake_rcst, ac_loss_fake_rcst, _ = obj_discriminator(
                        imgs_rcst_fake, objs, boxes, obj_to_img,
                        scatter_size_obj=scatter_size_obj)

                    d_obj_gan_rcst_loss = gan_d_loss(scores_real, scores_fake_rcst)
                    d_obj_losses.add_loss(d_obj_gan_rcst_loss, 'd_obj_gan_rcst_loss')
                    # d_obj_losses.add_loss(ac_loss_fake_rcst.mean(), 'd_ac_loss_fake_rcst')

                    optimizer_d_obj.zero_grad()
                    d_obj_losses.total_loss.backward()
                    optimizer_d_obj.step()

                if img_discriminator is not None:
                    d_img_losses = LossManager()
                    imgs_pred_fake = imgs_pred.detach()
                    imgs_rcst_fake = imgs_rcst.detach()

                    scores_real = img_discriminator(imgs)
                    scores_fake_pred = img_discriminator(imgs_pred_fake)
                    scores_fake_rcst = img_discriminator(imgs_rcst_fake)

                    d_img_gan_loss = gan_d_loss(scores_real, scores_fake_pred)
                    d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')

                    d_img_gan_rcst_loss = gan_d_loss(scores_real, scores_fake_rcst)
                    d_img_losses.add_loss(d_img_gan_rcst_loss, 'd_img_gan_rcst_loss')

                    optimizer_d_img.zero_grad()
                    d_img_losses.total_loss.backward()
                    optimizer_d_img.step()

            # Logging generative model losses
            for name, val in losses.items():
                logger.scalar_summary("loss/{}".format(name), val, t)
            # Logging discriminative model losses
            if obj_discriminator is not None:
                for name, val in d_obj_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            if img_discriminator is not None:
                for name, val in d_img_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            start_time = time.time()

        if epoch % args.eval_epochs == 0:

            log.info('[Epoch {}/{}] checking on val'.format(epoch, args.epochs))
            val_results = check_model(
                args, options, epoch, val_loader, model, vocab=vocab)
            val_losses, val_samples, val_batch_data, val_avg_iou, val_inception = val_results
            if val_inception[0] > best_inception[0]:
                isBest = True
                best_inception = val_inception
            checkpoint['counters']['best_inception'] = best_inception
            checkpoint['val_samples'].append(val_samples)
            checkpoint['val_batch_data'].append(val_batch_data)
            checkpoint['val_iou'].append(val_avg_iou)
            checkpoint['val_inception'].append(val_inception)
            logger.scalar_summary("ckpt/val_iou", val_avg_iou, epoch)
            for k, v in val_losses.items():
                checkpoint['val_losses'][k].append(v)
                logger.scalar_summary("ckpt/val_{}".format(k), v, epoch)
            logger.scalar_summary("ckpt/val_inception", val_inception[0], epoch)
            logger.image_summary(val_samples, epoch, tag="ckpt_val")
            log.info('[Epoch {}/{}] val iou: {}'.format(epoch, args.epochs, val_avg_iou))
            log.info('[Epoch {}/{}] val inception score: {} ({})'.format(
                    epoch, args.epochs, val_inception[0], val_inception[1]))
            log.info('[Epoch {}/{}] best inception scores: {} ({})'.format(
                    epoch, args.epochs, best_inception[0], best_inception[1]))


            checkpoint['model_state'] = model.module.state_dict()
            if obj_discriminator is not None:
                checkpoint['d_obj_state'] = obj_discriminator.module.state_dict()
                checkpoint['d_obj_optim_state'] = optimizer_d_obj.state_dict()

            if img_discriminator is not None:
                checkpoint['d_img_state'] = img_discriminator.module.state_dict()
                checkpoint['d_img_optim_state'] = optimizer_d_img.state_dict()

            checkpoint['optim_state'] = optimizer.state_dict()
            checkpoint['counters']['epoch'] = epoch
            checkpoint['counters']['t'] = t
            checkpoint['lr'] = lr
            checkpoint_path = os.path.join(log_path,
                                           '%s_with_model.pt' % args.checkpoint_name)
            log.info('[Epoch {}/{}] Saving checkpoint: {}'.format(epoch, args.epochs, checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            if isBest:
                copyfile(checkpoint_path, os.path.join(log_path, 'best_with_model.pt'))
                isBest = False


        if epoch >= args.decay_lr_epochs:
            lr_end = args.learning_rate * 1e-3
            decay_frac = (epoch - args.decay_lr_epochs + 1) / (args.epochs - args.decay_lr_epochs + 1e-5)
            lr = args.learning_rate - decay_frac * (args.learning_rate - lr_end)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            if img_discriminator is not None:
                for param_group in optimizer_d_img.param_groups:
                    param_group["lr"] = lr
            if obj_discriminator is not None:
                for param_group in optimizer_d_obj.param_groups:
                    param_group["lr"] = lr
            log.info('[Epoch {}/{}] learning rate: {}'.format(epoch+1, args.epochs, lr))

        logger.scalar_summary("ckpt/learning_rate", lr, epoch)

    # Evaluating after the whole training process.
    log.info("Evaluting the testing set.")
    val_mean, val_std = evaluate(model, val_loader, options)
    log.info("Inception score: {} ({})".format(val_mean, val_std))


if __name__ == '__main__':
    main()
