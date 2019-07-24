import torch
import torch.nn as nn
from .layers import GlobalAvgPool, build_cnn, get_activation, get_normalization_2d

class CropEncoder(nn.Module):
    '''
    Deterministic Crop Encoder:
    '''
    def __init__(self,
                 output_D,
                 num_categories=1,
                 cropEncoderArgs=None,
                 pooling="avg",
                 decoder_dims=None,):
        super(CropEncoder, self).__init__()
        # enable non-linear activation / batch normalization
        cropEncoderArgs["non_linear_activated"] = True
        cnn, cnn_D = build_cnn(**cropEncoderArgs)
        self.cnn_D = cnn_D
        self.model = cnn
        self.output_mean = nn.Sequential(
            nn.Conv2d(cnn_D, output_D * num_categories, kernel_size=1),
            get_normalization_2d(output_D, cropEncoderArgs["normalization"]),
            get_activation(cropEncoderArgs["activation"]),
        )
        self.output_D = output_D
        self.num_categories = num_categories

    def forward(self, x, objs):
        x = self.output_mean(self.model(x))
        if self.num_categories > 1:
            obj_cat = objs.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3)) \
                      * self.output_D + \
                      torch.arange(end=self.output_D,
                        device=objs.device).view(1, -1, 1, 1)
            x = x.gather(1, obj_cat)
        return x
