import json
import os
import os.path as osp
import numpy as np
from datasets import coco, visual_genome
from torch.utils.data import DataLoader


def build_coco_dsets(data_opts, batch_size, image_size, build_canvas=False):
    dset_kwargs = {
        'image_dir': osp.join(data_opts["root_dir"], data_opts["coco_train_image_dir"]),
        'instances_json': osp.join(data_opts["root_dir"], data_opts["coco_train_instances_json"]),
        'stuff_json': osp.join(data_opts["root_dir"], data_opts["coco_train_stuff_json"]),
        'stuff_only': data_opts["coco_stuff_only"],
        'image_size': data_opts["image_size"],
        'mask_size': data_opts["mask_size"],
        'max_samples': data_opts["num_train_samples"],
        'min_object_size': data_opts["min_object_size"],
        'min_objects_per_image': data_opts["min_objects_per_image"],
        'instance_whitelist': data_opts.get("instance_whitelist", None),
        'stuff_whitelist': data_opts.get("stuff_whitelist", None),
        'include_other': data_opts["coco_include_other"],
        'include_relationships': data_opts["include_relationships"],
        'use_object_crops': data_opts["use_object_crops"],
        'normalize_method': data_opts.get("normalize_method", "imagenet"),
    }
    if data_opts["use_object_crops"]:
        dset_kwargs_crops = {
            'mem_bank_path': osp.join(data_opts["root_dir"], data_opts["mem_bank"]),
            'crop_file_csv': osp.join(data_opts["root_dir"], data_opts["crop_file_csv"]),
            'top10_crop_ids': osp.join(data_opts["root_dir"], data_opts["top10_crop_ids"]),
            'object_index_mapping': osp.join(data_opts["root_dir"], data_opts["object_index_mapping"]),
            'crop_file_pickle': osp.join(data_opts["root_dir"], data_opts["crops_pickle"]),
            'crop_size': data_opts["crop_size"],
            'build_canvas': build_canvas,
            'retrieve_sampling': data_opts.get("retrieve_sampling", "random"),
            'crop_num': data_opts["crop_num"],
        }
        dset_kwargs = {**dset_kwargs, **dset_kwargs_crops}
    train_dset = coco(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d images and %d objects' %
          (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = osp.join(data_opts["root_dir"], data_opts["coco_val_image_dir"])
    dset_kwargs['instances_json'] = osp.join(data_opts["root_dir"], data_opts["coco_val_instances_json"])
    dset_kwargs['stuff_json'] = osp.join(data_opts["root_dir"], data_opts["coco_val_stuff_json"])
    dset_kwargs['max_samples'] = data_opts["num_val_samples"]
    # on validation set, we set it to a large number
    dset_kwargs['max_objects'] = 100
    dset_kwargs['object_index_mapping'] = osp.join(data_opts["root_dir"], data_opts["val_object_index_mapping"])
    dset_kwargs['top10_crop_ids'] = osp.join(data_opts["root_dir"], data_opts["val_top10_crop_ids"])
    val_dset = coco(**dset_kwargs)

    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))

    return vocab, train_dset, val_dset


def build_vg_dsets(data_opts, batch_size, image_size, build_canvas=False):
    with open(osp.join(data_opts["root_dir"], data_opts["vocab"]), 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
        'vocab': vocab,
        'h5_path': osp.join(data_opts["root_dir"], data_opts["trainset"]),
        'image_dir': osp.join(data_opts["root_dir"], data_opts["images"]),
        'image_size': image_size,
        'max_samples': data_opts["num_train_samples"],
        'max_objects': data_opts["max_objects_per_image"],
        'use_orphaned_objects': data_opts["use_orphaned_objects"],
        'use_object_crops': data_opts["use_object_crops"],
        'include_relationships': data_opts["include_relationships"],
        'normalize_method': data_opts.get("normalize_method", "imagenet"),
    }
    if data_opts["use_object_crops"]:
        dset_kwargs['mem_bank_path'] = osp.join(data_opts["root_dir"], data_opts["mem_bank"])
        dset_kwargs['crop_file_csv'] = osp.join(data_opts["root_dir"], data_opts["crops_csv"])
        dset_kwargs['crop_file_pickle'] = osp.join(data_opts["root_dir"], data_opts["crops_pickle"])
        dset_kwargs['top10_crop_ids'] = osp.join(data_opts["root_dir"], data_opts["top10_crop_ids"])
        dset_kwargs['build_canvas'] = build_canvas
        dset_kwargs['crop_size'] = data_opts.get("crop_size", None)
        dset_kwargs['retrieve_sampling'] = data_opts.get("retrieve_sampling", "random")
        dset_kwargs['crop_num'] = data_opts.get("crop_num", 1)

    train_dset = visual_genome(**dset_kwargs)
    iter_per_epoch = len(train_dset) // batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    # make some modification to dset_kwargs so that it can be used to create val_dset
    dset_kwargs['h5_path'] = osp.join(data_opts["root_dir"], data_opts["valset"])
    dset_kwargs['max_samples'] = data_opts["num_val_samples"]
    # on validation set, we set it to a large number
    dset_kwargs['max_objects'] = 100
    dset_kwargs['top10_crop_ids'] = osp.join(data_opts["root_dir"], data_opts["val_top10_crop_ids"])

    val_dset = visual_genome(**dset_kwargs)

    return vocab, train_dset, val_dset


def build_dataset(opts):
    if opts["dataset"] == "visual_genome":
        return build_vg_dsets(opts["data_opts"], opts["batch_size"],
                              opts["image_size"], opts.get("build_canvas", False))
    elif opts["dataset"] == "coco":
        return build_coco_dsets(opts["data_opts"], opts["batch_size"],
                                opts["image_size"], opts.get("build_canvas", False))
    else:
        raise ValueError("Unrecognized dataset: {}".format(opts["dataset"]))


def build_loaders(opts):
    vocab, train_dset, val_dset = build_dataset(opts)
    collate_fn = train_dset.collate_fn

    loader_kwargs = {
        'batch_size': opts["batch_size"],
        'num_workers': opts["workers"],
        'shuffle': True,
        'collate_fn': collate_fn,
        "drop_last": True,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    loader_kwargs['shuffle'] = False
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader
