import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.distributions as TD
import pickle


def logit2im(x):
    return x * 0.5 + 0.5


class MappingLayers(nn.Module):

    """
    Mapping Layers Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    """

    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            # A neural network which takes in tensors of
            # shape (n_samples, z_dim) and outputs (n_samples, w_dim)
            # with a hidden layer with hidden_dim neurons
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim),
        )

    def forward(self, noise):
        """
        Function for completing a forward pass of MappingLayers:
        Given an initial noise tensor, returns the intermediate noise tensor.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        return self.mapping(noise)


def scale_w(w, w_mean=None, w_weight=None):
    if w_mean is not None:
        w = w_mean + w_weight * (w - w_mean)
    return w


class InjectNoise(nn.Module):
    """
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    """

    def __init__(self, channels):

        super().__init__()
        self.weight = nn.Parameter(  # You use nn.Parameter so that these weights can be optimized
            # Initiate the weights for the channels from a random normal distribution
            torch.randn(channels)[None, :, None, None]  # torch.randn((1,channels,1,1))
        )

    def forward(self, image):
        """
        Function for completing a forward pass of InjectNoise: Given an image,
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        """
        # Set the appropriate shape for the noise!

        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])

        noise = torch.randn(noise_shape).to(image.device)  # Creates the random noise
        return (
            image + self.weight * noise
        )  # Applies to image after multiplying by the weight for each channel


# Adaptive Instance Norm
class AdaIN(nn.Module):
    """
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    """

    def __init__(self, channels, w_dim):
        super().__init__()

        # Normalize the input per-channels
        self.instance_norm = nn.InstanceNorm2d(channels)

        # You want to map w to a set of style weights per channel.
        # Replace the Nones with the correct dimensions - keep in mind that
        # both linear maps transform a w vector into style weights
        # corresponding to the number of image channels.
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        """
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w,
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector w
        """
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]

        # Calculate the transformed image
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image


class MicroStyleGANGeneratorBlock(nn.Module):
    """
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    """

    def __init__(
        self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True
    ):

        super().__init__()
        self.use_upsample = use_upsample

        if self.use_upsample:
            self.upsample = nn.Upsample(starting_size, mode="bilinear")
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size, padding=kernel_size // 2
        )  # Padding is used to maintain the image size
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        """
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w,
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        """
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.adain(x, w)
        return x


class StyledSequential(nn.Sequential):
    def forward(self, x, w):
        for module in self._modules.values():
            x = module(x, w)
        return x


class MicroStyleGANGenerator(nn.Module):
    """
    Micro StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    """

    def _sample_prior(self, n):
        x = self.prior.sample((n, self.z_dim)).to(self.starting_constant.device)
        return x

    def __init__(
        self,
        z_dim,  # z dimensionality
        map_hidden_dim,  # mapping network parameter
        w_dim,  # style vector dimensionality
        in_chan,  # number of channels in input trainable tensor
        out_chan,  # images dimensionality
        kernel_size,
        hidden_chan,
    ):
        super().__init__()
        self.prior = TD.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.z_dim = z_dim
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        # Typically this constant is initiated to all ones, but you will initiate to a
        # Gaussian to better visualize the network's effect
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))

        self.progression = nn.ModuleList(
            [
                StyledSequential(
                    *[
                        MicroStyleGANGeneratorBlock(
                            in_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            4,
                            use_upsample=False,
                        ),
                        MicroStyleGANGeneratorBlock(
                            hidden_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            4,
                            use_upsample=False,
                        ),
                    ]
                ),
                StyledSequential(
                    *[
                        MicroStyleGANGeneratorBlock(
                            hidden_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            8,
                            use_upsample=True,
                        ),
                        MicroStyleGANGeneratorBlock(
                            hidden_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            8,
                            use_upsample=False,
                        ),
                    ]
                ),
                StyledSequential(
                    *[
                        MicroStyleGANGeneratorBlock(
                            hidden_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            16,
                            use_upsample=True,
                        ),
                        MicroStyleGANGeneratorBlock(
                            hidden_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            16,
                            use_upsample=False,
                        ),
                    ]
                ),
                StyledSequential(
                    *[
                        MicroStyleGANGeneratorBlock(
                            hidden_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            32,
                            use_upsample=True,
                        ),
                        MicroStyleGANGeneratorBlock(
                            hidden_chan,
                            hidden_chan,
                            w_dim,
                            kernel_size,
                            32,
                            use_upsample=False,
                        ),
                    ]
                ),
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                nn.Sequential(
                    *[nn.Conv2d(hidden_chan, out_chan, kernel_size=1), nn.Tanh()]
                ),
                nn.Sequential(
                    *[nn.Conv2d(hidden_chan, out_chan, kernel_size=1), nn.Tanh()]
                ),
                nn.Sequential(
                    *[nn.Conv2d(hidden_chan, out_chan, kernel_size=1), nn.Tanh()]
                ),
                nn.Sequential(
                    *[nn.Conv2d(hidden_chan, out_chan, kernel_size=1), nn.Tanh()]
                ),
            ]
        )

    def forward(self, noise, step=0, alpha=-1, w_mean=None, w_weight=None):
        """
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise,
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
            return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
        """
        x = self.starting_constant
        w = self.map(noise)
        w = scale_w(w, w_mean, w_weight)

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):

            if i > 0 and step > 0:
                x_prev = x

            x = conv(x, w)

            if i == step:
                x = to_rgb(x)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](x_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode="bilinear")
                    x = (1 - alpha) * skip_rgb + alpha * x

                break

        return x

    def sample(self, n, step=3, alpha=-1, w_mean=None, w_weight=None):
        with torch.no_grad():
            return self.rsample(n, step, alpha, w_mean, w_weight)

    def rsample(self, n, step=3, alpha=-1, w_mean=None, w_weight=None):
        noise = self._sample_prior(n)
        return logit2im(self.forward(noise, step, alpha, w_mean, w_weight))
