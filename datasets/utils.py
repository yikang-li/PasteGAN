import PIL
import torch
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


STANDARD_MEAN = [0.5, 0.5, 0.5]
STANDARD_STD = [0.5, 0.5, 0.5]

INV_STANDARD_MEAN = [-m for m in STANDARD_MEAN]
INV_STANDARD_STD = [1.0 / s for s in STANDARD_STD]

def imagenet_preprocess(normalize_method='imagenet'):
    if normalize_method == 'imagenet':
        return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif normalize_method == 'standard':
        return T.Normalize(mean=STANDARD_MEAN, std=STANDARD_STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo + 1e-5)


def imagenet_deprocess(rescale_image=True, normalize_method='imagenet'):
    if normalize_method == 'imagenet':
        transforms = [
            T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
            T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
        ]
    elif normalize_method == 'standard':
        transforms = [
            T.Normalize(mean=[0, 0, 0], std=INV_STANDARD_STD),
            T.Normalize(mean=INV_STANDARD_MEAN, std=[1.0, 1.0, 1.0]),
        ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=False, normalize_method='imagenet'):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess(rescale_image=rescale, normalize_method=normalize_method)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255).byte()
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de


def unpack_var(v):
    if isinstance(v, torch.autograd.Variable):
        return v.data
    return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
    triples = unpack_var(triples)
    obj_data = [unpack_var(o) for o in obj_data]
    obj_to_img = unpack_var(obj_to_img)
    triple_to_img = unpack_var(triple_to_img)

    triples_out = []
    obj_data_out = [[] for _ in obj_data]
    obj_offset = 0
    N = obj_to_img.max() + 1
    for i in range(N):
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)

        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        triples_out.append(cur_triples)

        for j, o_data in enumerate(obj_data):
            cur_o_data = None
            if o_data is not None:
                cur_o_data = o_data[o_idxs]
            obj_data_out[j].append(cur_o_data)

        obj_offset += o_idxs.size(0)

    return triples_out, obj_data_out
