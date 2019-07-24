"""
Functions used to making canvas from object crops / object feature crops
"""

import torch
import torch.nn.functional as F
from .layout import _boxes_to_grid




def _get_samples(object_crops, boxes, H, W=None):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space
    - H, W: Size of the output

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    # If we don't add extra spatial dimensions here then out-of-bounds
    # elements won't be automatically set to 0
    samples = F.grid_sample(object_crops, grid)   # (O, D, H, W)

    return samples

def _get_int_boxes(boxes, H, W=None):
    """
    Transform the boxes [x0, y0, x1, y1] in the [0, 1] coordinates to the
    integer coordinates in the [0, H) and [0, W).

    input:
     - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
       [x0, y0, x1, y1] in the [0, 1] coordinate space. All the boxes belong
       to the same image
     - H, W: the size of the output (integer)

    Output:
     - int_boxes: LongTensor of shape (O, 4) giving the bounding boxes in the
       format [x0, y0, x1, y1]

    """
    if W is None:
        W = H
    # avoid in-place operation
    int_boxes = torch.zeros_like(boxes)
    int_boxes[:, 0] = boxes[:, 0] * W
    int_boxes[:, 2] = boxes[:, 2] * W
    int_boxes[:, 1] = boxes[:, 1] * H
    int_boxes[:, 3] = boxes[:, 3] * H
    return int_boxes.long()

def make_canvas_baseline(boxes, object_crops, H, W=None, transparent_mask=False):
    """
    Baseline method to build canvas:
        We sort the object crops accoding to its area. Put the crops with larger
        area to the back.

    input:
     - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
       [x0, y0, x1, y1] in the [0, 1] coordinate space. All the boxes belong
       to the same image
     - object_crops: Tensor of the shape (O, 3, H, W) giving the retrieved
       object crops
     - H, W: the size of the output

    Output:
     - canvas: Tensor of the shape (3, H, W) giving the base canvas which
       we use to generate the reconstructed image.
    """
    if W is None:
        W = H

    Ws = boxes[:, 2] - boxes[:, 0]
    Hs = boxes[:, 3] - boxes[:, 1]
    Areas = Ws * Hs

    _, ids = torch.sort(Areas, descending=True)
    int_boxes = _get_int_boxes(boxes, H, W)
    samples = _get_samples(object_crops, boxes, H, W)

    canvas = torch.zeros(3, H, W)
    if not transparent_mask:
        for id in ids:
            canvas[:,
                   int_boxes[id, 1]: int_boxes[id, 3],
                   int_boxes[id, 0]: int_boxes[id, 2]] = \
                   samples[id, :,
                        int_boxes[id, 1]: int_boxes[id, 3],
                        int_boxes[id, 0]: int_boxes[id, 2]]
    if transparent_mask:
        '''
        # too show
        for id in ids:
            for y in range(int_boxes[id, 1], int_boxes[id, 3]):
                for x in range(int_boxes[id, 0], int_boxes[id, 2]):
                    if not (float(samples[id, 0, y, x].item()) == 0.0 and
                            float(samples[id, 1, y, x].item()) == 0.0 and
                            float(samples[id, 2, y, x].item()) == 0.0) :
                        canvas[:, y, x] = samples[id, :, y, x]
        '''
        # fast approach.
        for id in ids:
            shade_mask = samples[id] != 0.
            canvas[shade_mask] = samples[id, shade_mask]

    return canvas


if __name__ == '__main__':
    boxes = torch.Tensor([
            [0., 0.2, 0.6, 0.6],
            [0., 0., 1., 1.],
    ])
    object_crops = torch.randn(2, 3, 96, 96)
    H = 128

    result = make_canvas_baseline(boxes, object_crops, H)

    print("Output Size: ", result.size())
