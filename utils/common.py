import os
import time
import inspect
import subprocess
from contextlib import contextmanager

import torch
from PIL import Image
import numpy as np
from collections import OrderedDict

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def float_tuple(s):
    return tuple(float(i) for i in s.split(','))


def str_tuple(s):
    return tuple(s.split(','))


def set_trainable(model, requires_grad):
    set_trainable_param(model.parameters(), requires_grad)

def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict) and key in dict_to.keys():
            update_values(dict_from[key], dict_to[key])
        elif value is not None or key not in dict_to.keys():
            dict_to[key] = dict_from[key]
    return dict_to


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return inspect.currentframe().f_back.f_lineno


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(0), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[1].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, torch.Tensor):
        return input_image
    image_numpy = input_image.numpy()
    if image_numpy.ndim == 3:
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        elif image_numpy.shape[0] == 3:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
    elif image_numpy.ndim == 4:
        if image_numpy.shape[1] == 1:
            image_numpy = np.tile(image_numpy, (1, 3, 1, 1))
        elif image_numpy.shape[1] == 3:
            image_numpy = np.transpose(image_numpy, (0, 2, 3, 1))
    else:
        raise ValueError("Only support images or batches")
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


class LossManager(object):
    def __init__(self):
        self.total_loss = None
        self.all_losses = {}

    def add_loss(self, loss, name, weight=1.0):
        cur_loss = loss * weight
        if self.total_loss is not None:
            self.total_loss += cur_loss
        else:
            self.total_loss = cur_loss

        self.all_losses[name] = cur_loss.data.cpu().item()

    def items(self):
        return self.all_losses.items()


def load_model_state(model, state_dict, strict=True):
    # sometimes if the state_dict is saved from DataParallel model, the keys
    # has a prefix "module.", so we remove them before Loading
    new_state_dict = OrderedDict()
    if not isinstance(model, torch.nn.DataParallel):
        for k, v in state_dict.items():
            k = k[7:] if k.startswith("module") else k
            new_state_dict[k] = v
    else:
        new_state_dict = state_dict
    if strict:
        # requires the model's state_dict to be exactly same as the new_state_dict
        model.load_state_dict(new_state_dict)
    else:
        # only load parameters that are same size and type, and print warnings
        # for parameters that don't match
        own_state = model.state_dict()
        for name, param in new_state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                if own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                else:
                    print(
                        "Warning: While copying the parameter named {}, "
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}.".format(
                            name, own_state[name].size(), param.size()
                        )
                    )
            else:
                print(
                    "Warning: Parameter named {} is not used "
                    "by this model.".format(name)
                )
        for name, _ in own_state.items():
            if name not in new_state_dict:
                print(
                    "Warning: Parameter named {} in the model "
                    "is not initialized.".format(name)
                )
    return model
