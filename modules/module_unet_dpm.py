import os
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from networks.unet import Generator, Discriminator
from utils.batch_utils_dpm import Dataset_train, Dataset_test
from utils.loss import SSIM_loss, GANLoss, TVLoss
from PIL import Image


class GAN(pl.LightningModule):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.save_hyperparameters("config")
        self.config = config
        self.generator = self.defineG()
        self.discriminator = self.defineD()
        self.criterionGAN = GANLoss(use_lsgan=False)
        self.criterionL1 = nn.SmoothL1Loss(reduction="mean")
        self.criterionSSIM = SSIM_loss(size_average=True)
        self.criterionTV = TVLoss()

        # cache for generated images
        self.last_inps = None
        self.last_imgs = None
        self.last_lbls = None
        # self.curr_epoch = 0

    def forward(self, img):
        return self.generator(img)

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

    def validation_step(self, batch, batch_nb):
        input_image, real_image = batch
        g_losses, generated = self.compute_generator_loss(input_image, real_image)
        if batch_nb == 0:
            img_save_num = min(20, input_image.shape[0])
            inp_channel_num = min(3, input_image.shape[1])
            self.last_inps = input_image[:img_save_num, [0,0,0], :, :]
            self.last_imgs = generated[:img_save_num, :, :, :]
            self.last_lbls = real_image[:img_save_num, :, :, :]
        self.log_dict(
            {key + "/valid": g_losses[key] for key in g_losses.keys()},
            on_epoch=True,
            on_step=False,
        )

    def validation_epoch_end(self, outputs):
        tb_logger = self.logger

        input_grid = torchvision.utils.make_grid(
            self.last_inps, nrow=1, normalize=True, value_range=(0, 10)
        )
        output_grid = torchvision.utils.make_grid(
            self.last_imgs, nrow=1, normalize=True, value_range=(-1, 1)
        )
        label_grid = torchvision.utils.make_grid(
            self.last_lbls, nrow=1, normalize=True, value_range=(-1, 1)
        )

        tb_logger.experiment.add_image(
            "valid",
            torch.cat([input_grid, output_grid, label_grid], dim=2),
            self.current_epoch,
        )

    def test_step(self, batch, batch_nb):
        input_images, image_names = batch
        # if not os.path.exists(self.config.output_images_dir + "/" + image_names[0] + ".png"):
        generated = self.generate_fake(input_images)
        for i, fname in enumerate(image_names):
            img = self.norm_output(generated[i])
            img.save(self.config.output_images_dir + "/" + fname + ".png")

    def compute_generator_loss(self, input_image, real_image):
        G_losses = {}
        fake_image = self.generate_fake(input_image)
        pred_fake, _ = self.discriminate(input_image, fake_image, real_image)
        if self.config.lambda_dis > 0:
            G_losses["GAN"] = (
                self.criterionGAN(pred_fake, True) * self.config.lambda_dis
            )
        if self.config.lambda_l1 > 0:
            G_losses["L1"] = (
                self.criterionL1(fake_image, real_image) * self.config.lambda_l1
            )
        if self.config.lambda_ssim > 0:
            G_losses["SSIM"] = (
                self.criterionSSIM((fake_image + 1.0) / 2.0, (real_image + 1.0) / 2.0)
                * self.config.lambda_ssim
            )
        if self.config.lambda_tv > 0:
            G_losses["TV"] = self.criterionTV(fake_image) * self.config.lambda_tv
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_image, fake_image, real_image):
        D_losses = {}
        pred_fake, pred_real = self.discriminate(input_image, fake_image, real_image)
        D_losses["D_Fake"] = self.criterionGAN(pred_fake, False)
        D_losses["D_real"] = self.criterionGAN(pred_real, True)
        return D_losses

    def generate_fake(self, input_image):
        input_image = torch.clip(input_image, -20, 20)
        fake_image = self.generator(input_image)
        return fake_image

    def discriminate(self, input_image, fake_image, real_image):
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        if self.config.conditional is True:
            input_image = torch.clip(input_image, -20, 20)
            input_and_input = torch.cat([input_image, input_image], dim=0)
            discriminator_out = self.discriminator(
                torch.cat([input_and_input, fake_and_real], 1)
            )
        else:
            discriminator_out = self.discriminator(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2 :]

        return fake, real

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
        return [
            {"optimizer": opt_g, "frequency": self.config.opt_g_freq},
            {"optimizer": opt_d, "frequency": self.config.opt_d_freq},
        ]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def train_dataloader(self):
        tds = Dataset_train(
            self.config.train_images_dir,
            self.config.crop_size,
            is_training=True,
            color_space=self.config.color_space,
            n_workers=self.config.n_threads,
            epoch_len=(
                self.config.epoch_len * self.config.batch_size * self.config.ngpu
            ),
            queue_size=self.config.data_queue_len,
            patch_per_tile=self.config.patch_per_tile,
            raw_downsample=self.config.raw_downsample,
        )
        loader = torch.utils.data.DataLoader(
            dataset=tds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        vds = Dataset_train(
            self.config.valid_images_dir,
            self.config.crop_size,
            is_training=False,
            color_space=self.config.color_space,
            n_workers=self.config.n_threads,
            epoch_len=(
                (self.config.epoch_len // 10 + 1)
                * self.config.batch_size
                * self.config.ngpu
            ),
            queue_size=(
                self.config.data_queue_len // 10
                + self.config.batch_size * self.config.ngpu
            ),
            patch_per_tile=self.config.patch_per_tile,
            raw_downsample=self.config.raw_downsample,
        )
        loader = torch.utils.data.DataLoader(
            dataset=vds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        tds = Dataset_test(self.config.test_images_dir)
        loader = torch.utils.data.DataLoader(
            dataset=tds,
            batch_size=self.config.batch_size_test,
            shuffle=False,
            num_workers=self.config.n_workers_test,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def defineG(self):
        netG = Generator(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            n_levels=self.config.n_blocks,
            n_channels=self.config.n_channels,
            padding=True,
            batch_norm=self.config.batch_norm,
            up_mode="upsample",
        )
        return netG

    def defineD(self):
        if self.config.conditional is True:
            in_channels_D = self.config.out_channels + self.config.in_channels
        else:
            in_channels_D = self.config.out_channels
        netD = Discriminator(
            in_channels=in_channels_D,
            n_levels=self.config.n_blocks,
            n_channels=self.config.n_channels,
            image_size=self.config.crop_size,
            padding=True,
            batch_norm=self.config.batch_norm,
        )
        return netD

    def norm_output(self, img):
        img.clone()
        img.clamp_(min=-1, max=1)
        img.add_(1).div_(2)
        ndarr = (
            img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy().squeeze()
        )
        im = Image.fromarray(ndarr)
        return im

    def modify_params(self, config):
        for key in config.keys():
            self.config[key] = config[key]
        if self.config.is_testing:
            if not os.path.exists(self.config.output_images_dir):
                os.makedirs(self.config.output_images_dir)
