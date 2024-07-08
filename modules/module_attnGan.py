import torch
from networks.attention_gan import Generator, Discriminator
from modules.module_unet import GAN as UnetGan

# Define the custom GAN class inheriting from UnetGan
class GAN(UnetGan):
    def __init__(self, config):
        super().__init__(config)  # Initialize the superclass with the given configuration

    def defineG(self):
        # Define the generator network
        netG = Generator(
            self.config,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            padding=1,
            batch_norm=self.config.batch_norm,
            pooling_mode="maxpool",
        )
        # Optionally initialize weights (commented out)
        # netG.apply(weights_init)
        return netG

    def defineD(self):
        # Determine the input channels for the discriminator
        if self.config.conditional is True:
            in_channels_D = self.config.out_channels + self.config.in_channels
        else:
            in_channels_D = self.config.out_channels
        
        # Define the discriminator network
        netD = Discriminator(
            self.config, in_channels=in_channels_D, batch_norm=self.config.batch_norm
        )
        # Optionally initialize weights (commented out)
        # netD.apply(weights_init)
        return netD

    def configure_optimizers(self):
        # Define optimizers for the generator and discriminator
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr_g,
            betas=(self.config.beta1, self.config.beta2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr_d,
            betas=(self.config.beta1, self.config.beta2),
        )
        
        # Define a learning rate scheduler for the generator
        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, factor=0.5, patience=20, threshold=0.0001, min_lr=1e-8, verbose=True
        )
        my_lr_schd = {
            "scheduler": scheduler_g,
            "interval": "epoch",
            "frequency": 1,
            "reduce_on_plateau": True,
            "monitor": "L1/valid",
            "strict": True,
            "name": None,
        }
        
        # Return the optimizers and the learning rate scheduler
        return [opt_g, opt_d], [my_lr_schd]