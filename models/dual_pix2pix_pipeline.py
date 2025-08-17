# dual_pix2pix_pipeline.py
import torch
from torch import nn
import itertools

class UNetDown(nn.Module):
    """A downsampling block in the U-Net Generator."""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """An upsampling block in the U-Net Generator."""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    """The U-Net Generator architecture for image-to-image translation."""
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        # Bottleneck
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final_up(u5)


class Discriminator(nn.Module):
    """The PatchGAN Discriminator to classify image patches as real or fake."""
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class DualPix2PixPipeline(nn.Module):
    """
    Manages two Generator-Discriminator pairs for forward (t -> t+1)
    and backward (t+1 -> t) image translation, with cycle consistency.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Forward: x_t -> x_t1
        self.forward_generator = Generator(in_channels, out_channels)
        self.forward_discriminator = Discriminator(in_channels)

        # Backward: x_t1 -> x_t
        self.backward_generator = Generator(in_channels, out_channels)
        self.backward_discriminator = Discriminator(in_channels)

        # Loss functions
        self.adversarial_loss = nn.MSELoss()  # LSGAN loss
        self.reconstruction_loss = nn.L1Loss() # L1 loss for image generation
        self.cycle_loss = nn.L1Loss() # L1 loss for cycle consistency

    def get_generator_params(self):
        """Returns parameters for both generators for the optimizer."""
        return itertools.chain(self.forward_generator.parameters(), self.backward_generator.parameters())

    def get_discriminator_params(self):
        """Returns parameters for both discriminators for the optimizer."""
        return itertools.chain(self.forward_discriminator.parameters(), self.backward_discriminator.parameters())