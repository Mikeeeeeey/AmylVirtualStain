import torch
import torch.nn as nn
from utils.pytorch_ssim import SSIM

# Total Variation Loss to encourage smoothness in images
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight  # Weight for the TV loss component

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# GAN Loss for training the discriminator and generator
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()  # LSGAN uses MSE loss
        else:
            self.loss = nn.BCEWithLogitsLoss()  # Standard GAN uses BCE loss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# SSIM Loss for measuring structural similarity between images
class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        self.loss_fun = SSIM(window_size=window_size, size_average=size_average, val_range=1)

    def forward(self, img1, img2):
        return -torch.log((1 + self.loss_fun(img1, img2)) / 2)
        # Alternatively, without the log transformation:
        # return 1 - (1 + self.loss_fun(img1, img2))/2
