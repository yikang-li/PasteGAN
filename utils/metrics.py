import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def intersection(bbox_pred, bbox_gt):
    max_xy = torch.min(bbox_pred[:, 2:], bbox_gt[:, 2:])
    min_xy = torch.max(bbox_pred[:, :2], bbox_gt[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def jaccard(bbox_pred, bbox_gt):
    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] -
                                                       bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] -
                                                 bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union)
    return torch.sum(iou)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
                total_norm = total_norm ** (1. / norm_type)
            except:
                continue
    return total_norm
