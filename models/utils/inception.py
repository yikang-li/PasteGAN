import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


class InceptionVisualExtracter(nn.Module):
    def __init__(self, pretrained=True, avg_pool=False):
        super(InceptionVisualExtracter, self).__init__()

        self.avg_pool = avg_pool
        self.inception = torchvision.models.inception_v3(pretrained=pretrained)

        for param in self.inception.parameters():
            param.requires_grad = False

        self.Conv2d_1a_3x3 = self.inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = self.inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = self.inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = self.inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = self.inception.Conv2d_4a_3x3
        self.Mixed_5b = self.inception.Mixed_5b
        self.Mixed_5c = self.inception.Mixed_5c
        self.Mixed_5d = self.inception.Mixed_5d
        self.Mixed_6a = self.inception.Mixed_6a
        self.Mixed_6b = self.inception.Mixed_6b
        self.Mixed_6c = self.inception.Mixed_6c
        self.Mixed_6d = self.inception.Mixed_6d
        self.Mixed_6e = self.inception.Mixed_6e
        self.Mixed_7a = self.inception.Mixed_7a
        self.Mixed_7b = self.inception.Mixed_7b
        self.Mixed_7c = self.inception.Mixed_7c


    def forward(self, imgs, **kwargs,):
        x = F.interpolate(imgs, (299, 299), mode="bilinear")
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # sub-region features is the output of Mixed_6e
        sub_region_features = x

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        if not self.avg_pool:
            # 8 x 8 x 2048
            return x
        else:
            # 1 x 1 x 2048
            return F.avg_pool2d(x, kernel_size=8)
