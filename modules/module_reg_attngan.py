import torch
from networks.attention_gan import Generator, Discriminator
from modules.module_unet import GAN as UnetGan
from utils.reggan.transformer import Transformer_2D
from utils.reggan.reg import Reg


class GAN(UnetGan):
    def __init__(self, config):
        super().__init__(config)
        self.R_A = self.defineR()
        self.spatial_transform = Transformer_2D()

    def defineG(self):
        netG = Generator(
            self.config,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            padding=1,
            batch_norm=self.config.batch_norm,
            pooling_mode="maxpool",
        )
        return netG

    def defineD(self):
        if self.config.conditional is True:
            in_channels_D = self.config.out_channels + self.config.in_channels
        else:
            in_channels_D = self.config.out_channels
        netD = Discriminator(
            self.config,
            in_channels=in_channels_D,
            padding=1,
            batch_norm=self.config.batch_norm,
        )
        return netD

    def defineR(self):
        netR = Reg(
            self.config.crop_size,
            self.config.crop_size,
            self.config.out_channels,
            self.config.out_channels,
        )
        return netR

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.lr_g,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.lr_d,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )
        opt_r = torch.optim.Adam(
            self.R_A.parameters(), lr=self.config.lr_r, betas=(0.5, 0.999)
        )
        return [
            {"optimizer": opt_g, "frequency": self.config.opt_g_freq},
            {"optimizer": opt_d, "frequency": self.config.opt_d_freq},
            {"optimizer": opt_r, "frequency": self.config.opt_r_freq},
        ]

    def training_step(self, batch, batch_nb, optimizer_idx):
        input_image, real_image = batch

        # train generator
        if optimizer_idx == 0:
            # generate images
            g_losses, _ = self.compute_generator_loss(input_image, real_image)
            self.g_losses = g_losses
            # self.generated = generated.detach()
            self.log_dict(
                {key + "/train": g_losses[key] for key in g_losses.keys()},
                on_step=False,
                on_epoch=True,
            )
            g_loss = sum(g_losses.values()).mean()
            return g_loss

        if optimizer_idx == 1:
            fake_image = self.generate_fake(input_image).detach()
            d_losses = self.compute_discriminator_loss(
                input_image, fake_image, real_image
            )
            self.d_losses = d_losses
            self.log_dict(
                {key + "/train": d_losses[key] for key in d_losses.keys()},
                on_step=False,
                on_epoch=True,
            )
            d_loss = sum(d_losses.values()).mean()
            return d_loss

        if optimizer_idx == 2:
            fake_image = self.generate_fake(input_image).detach()
            real_image, trans = self.correct_reg(fake_image, real_image)
            r_losses = {}
            r_losses["REG"] = (
                self.criterionL1(fake_image, real_image) * self.config.lambda_reg
            )
            # r_losses["SMTH"] = self.smooothing_loss(trans) * self.config.lambda_smth
            self.log_dict(
                {key + "/train": r_losses[key] for key in r_losses.keys()},
                on_step=False,
                on_epoch=True,
            )
            r_loss = sum(r_losses.values()).mean()
            return r_loss

    def compute_generator_loss(self, input_image, real_image):
        G_losses = {}
        fake_image = self.generate_fake(input_image)
        pred_fake, _ = self.discriminate(input_image, fake_image, real_image)
        if self.config.lambda_dis > 0:
            G_losses["GAN"] = (
                self.criterionGAN(pred_fake, True) * self.config.lambda_dis
            )
        real_image, _ = self.correct_reg(fake_image, real_image)
        if self.config.lambda_l1 > 0:
            G_losses["L1"] = (
                self.criterionL1(fake_image, real_image) * self.config.lambda_l1
            )
        if self.config.lambda_ssim > 0:
            G_losses["SSIM"] = (
                self.criterionSSIM((fake_image + 1.0) / 2.0, (real_image + 1.0) / 2.0)
                * self.config.lambda_ssim
            )
        return G_losses, fake_image

    def correct_reg(self, output_image, label_image):
        Trans = self.R_A(label_image, output_image)
        SysRegist_A2B = self.spatial_transform(label_image, Trans)
        return SysRegist_A2B, Trans

    def smooothing_loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        dx = dx * dx
        dy = dy * dy
        d = torch.mean(dx) + torch.mean(dy)
        return d
