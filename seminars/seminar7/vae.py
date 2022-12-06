import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################
# Taken from the hw2 solutions!

def get_normal_nll(x, mean, log_std):
    """
        This function should return the negative log likelihood log p(x),
        where p(x) = Normal(x | mean, exp(log_std) ** 2).
        Note that we consider the case of diagonal covariance matrix.
    """
    return 0.5 * np.log(2 * np.pi) + log_std + (x - mean) ** 2 * torch.exp(-2 * log_std) * 0.5

def get_normal_KL(mean_1, log_std_1, mean_2=None, log_std_2=None):
    """
        :Parameters:
        mean_1: means of normal distributions (1)
        log_std_1 : standard deviations of normal distributions (1)
        mean_2: means of normal distributions (2)
        log_std_2 : standard deviations of normal distributions (2)
        :Outputs:
        kl divergence of the normal distributions (1) and normal distributions (2)
        ---
        This function should return the value of KL(p1 || p2),
        where p1 = Normal(mean_1, exp(log_std_1) ** 2), p2 = Normal(mean_2, exp(log_std_2) ** 2).
        If mean_2 and log_std_2 are None values, we will use standard normal distribution.
        Note that we consider the case of diagonal covariance matrix.
    """
    if mean_2 is None:
        mean_2 = torch.zeros_like(mean_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)
    assert mean_1.shape == log_std_1.shape == mean_2.shape == log_std_2.shape
    return (log_std_2 - log_std_1) + (torch.exp(log_std_1 * 2) + (mean_1 - mean_2) ** 2) / 2 / torch.exp(log_std_2 * 2) - 0.5

class FullyConnectedMLP(nn.Module):
    def __init__(self, input_shape, hiddens, output_shape):
        assert isinstance(hiddens, list)
        super().__init__()
        self.input_shape = (input_shape,)
        self.output_shape = (output_shape,)
        self.hiddens = hiddens

        model = []
        prev_h = input_shape
        for h in hiddens:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.append(nn.Linear(hiddens[-1], output_shape))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, *self.output_shape)

class VAE2d(nn.Module):

    def __init__(self, n_in, n_latent, enc_hidden_sizes, dec_hidden_sizes):
        assert isinstance(enc_hidden_sizes, list)
        assert isinstance(dec_hidden_sizes, list)
        super().__init__()
        self.n_latent = n_latent
        self.encoder = FullyConnectedMLP(n_in, enc_hidden_sizes, 2 * n_latent)
        self.decoder = FullyConnectedMLP(n_latent, dec_hidden_sizes, 2 * n_in)

    def prior(self, n):
        print('here')
        return torch.randn(n, self.n_latent).to(next(self.parameters()))

    def forward(self, x):
        mu_z, log_std_z = self.encoder(x).chunk(2, dim=1)
        z = self.prior(mu_z.shape[0]) * log_std_z.exp() + mu_z
        mu_x, log_std_x = self.decoder(z).chunk(2, dim=1)
        return mu_z, log_std_z, mu_x, log_std_x

    def loss(self, x):
        mu_z, log_std_z, mu_x, log_std_x = self(x)
        recon_loss = get_normal_nll(x, mu_x, log_std_x)
        recon_loss = recon_loss.sum(1).mean()

        kl_loss = get_normal_KL(mu_z, log_std_z)
        kl_loss = kl_loss.sum(1).mean()

        return {
            'elbo_loss': recon_loss + kl_loss, 
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def sample(self, n, sample_from_decoder=True):
        with torch.no_grad():
            z = self.prior(n)
            mu, log_std = self.decoder(z).chunk(2, dim=1)
            if sample_from_decoder:
                z = torch.randn_like(mu) * log_std.exp() + mu
            else:
                z = mu
            # ====
        return z.cpu().numpy()