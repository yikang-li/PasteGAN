import torch
import models
from copy import deepcopy
from utils import update_values, load_model_state
import glog as log

def build_model(opts, vocab, image_size, checkpoint_start_from=None):
    if checkpoint_start_from is not None:
        log.info("Load checkpoint as initialization: {}".format(checkpoint_start_from))
        checkpoint = torch.load(checkpoint_start_from)
        # kwarg aka keyword arguments
        kwargs = checkpoint['model_kwargs']
        model = getattr(models, opts["arch"])(**kwargs)
        raw_state_dict = checkpoint['model_state']
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model = load_model_state(model, state_dict, strict=False)
    else:
        kwargs = deepcopy(opts["options"])
        kwargs["vocab"] = vocab
        kwargs["image_size"] = image_size
        model = getattr(models, opts["arch"])(**kwargs)
    return model, kwargs

def build_obj_discriminator(opts, vocab):
    d_kwargs = deepcopy(opts["generic"])
    d_kwargs = update_values(opts["object"], d_kwargs)
    d_kwargs["vocab"] = vocab
    discriminator = models.AcCropDiscriminator(**d_kwargs)
    return discriminator, d_kwargs

def build_img_discriminator(opts, vocab):
    d_kwargs = deepcopy(opts["generic"])
    d_kwargs = update_values(opts["image"], d_kwargs)
    discriminator = models.PatchDiscriminator(**d_kwargs)
    return discriminator, d_kwargs
