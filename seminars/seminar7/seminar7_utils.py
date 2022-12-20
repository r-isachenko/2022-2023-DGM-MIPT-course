import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torch.distributions as TD
from PIL import Image
import itertools
import pandas as pd

import sys
sys.path.append('../../homeworks') 
from dgm_utils import train_model, show_samples
from dgm_utils import visualize_2d_samples, visualize_2d_densities, visualize_2d_data
from dgm_utils.visualize import (
    TICKS_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LABEL_FONT_SIZE,
    TITLE_FONT_SIZE)

def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

##############
# Taken from the seminar3

class BananaDistribution:

    def __init__(self, inv=False, device='cpu'):
        self.inv = inv
        self.x2_distrib = TD.Normal(
            torch.tensor(0.0).to(device), 
            torch.tensor(3.3).to(device))
        self.x1_distrib = TD.Normal(
            torch.tensor(0.0).to(device),
            torch.tensor(1.).to(device))

    def sample(self, count):
        if not isinstance(count, tuple):
            count = (count,)
        x2 = self.x2_distrib.sample(count)
        x1 = self.x1_distrib.sample(count) + (x2**2)/8.
        samples = torch.stack([x1, x2], axis=-1)
        if self.inv:
            samples = samples.flip(-1)
        return samples
    
    def log_prob(self, samples):
        if self.inv:
            samples = samples.flip(-1)
        x2 = samples[..., 1]
        x1 = samples[..., 0]
        log_prob_x2 = self.x2_distrib.log_prob(x2)
        log_prob_x1 = self.x1_distrib.log_prob(x1 - (x2**2)/8.)
        return log_prob_x2 + log_prob_x1
    
    def prob(self, samples):
        return torch.exp(self.log_prob(samples))

class MaskedLinear(torch.nn.Linear):
    
    
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
         
        return torch.nn.functional.linear(input, self.mask * self.weight, self.bias)


class MADE(torch.nn.Module):
    
    def __init__(self, dim, hidden_sizes, out_bins, in_bins=1):
        '''
        :Parameters:
        dim : int : number of input dimensions
        hidden_sizes : list : sizes of hidden layers
        out_bins : output params per each output dimension
        in_bins : input params per each input dimension (for example, one hot)
        '''
        super().__init__()
        
        self.dim = dim
        self.nin = dim * in_bins
        self.in_bins = in_bins
        self.nout = dim * out_bins
        self.out_bins = out_bins

        self.hidden_sizes = hidden_sizes
        # we will use the trivial ordering of input units
        self.ordering = np.arange(self.dim)

        self.net = []
        hs = [self.nin, ] + self.hidden_sizes + [self.nout, ]
        for h0, h1 in zip(hs[:-2], hs[1:-1]):
            self.net.extend([
                MaskedLinear(h0, h1),
                torch.nn.ReLU(),
            ])

        self.net.append(MaskedLinear(hs[-2], hs[-1]))
        self.net = torch.nn.Sequential(*self.net)

        self.create_mask()  # builds the initial self.m connectivity


    def create_mask(self):
        
        # 1) The ordering of input units from 1 to d (self.ordering).
        # 2) Assign the random number k from 1 to d − 1 to each hidden unit. 
        #    This number gives the maximum number of input units to which the unit can be connected.
        # 3) Each hidden unit with number k is connected with the previous layer units 
        #   which has the number less or equal than k.
        # 4) Each output unit with number k is connected with the previous layer units 
        #    which has the number less than k.

        self.assigned_numbers = {}
        self.masks = []
        L = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.assigned_numbers[-1] = self.ordering
        for l in range(L):
            self.assigned_numbers[l] = np.random.randint(
                self.assigned_numbers[l - 1].min(), self.dim - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.assigned_numbers[l - 1][:, None] <= self.assigned_numbers[l][None, :] for l in range(L)]
        masks.append(self.assigned_numbers[L - 1][:, None] < self.assigned_numbers[-1][None, :])

        masks[-1] = np.repeat(masks[-1], self.out_bins, axis=1)
        masks[0] = np.repeat(masks[0], self.in_bins, axis=0)
        self.masks = masks 

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)


    def visualize_masks(self):
        prod = self.masks[0]
        for idx, m in enumerate(self.masks):
            plt.figure(figsize=(3, 3))
            plt.title(f'layer: {idx}')
            plt.imshow(m.T, vmin=0, vmax=1, cmap='gray')
            plt.show()

            if idx > 0:
                prod=prod.dot(m)

        plt.figure(figsize=(3, 3))
        plt.title('prod')
        plt.imshow(prod.T, vmin=0, vmax=1, cmap='gray')
        plt.show()


    def forward(self, x):
        """
        :Parameters:
        x: torch.Size([BS, nin]) : input sample
        :Output:
        out : torch.Size([BS, nout]) : output 
        """
        assert len(x.size()) == 2
        assert x.shape[1] == self.nin
        batch_size = x.shape[0]
        logits = self.net(x)
        return logits

def generate_2d_image_data(count, bins=64):
    # Загружаем картинку, сжимаем к размеру (bins x bins),
    # конвертируем к grayscale - формату
    im = Image.open('pics/2d_distribution.png').resize((bins, bins)).convert('L')
    im = np.array(im).astype('float32')
    # Сейчас im : np.array размера (64, 64), 
    # элементы этого массива выглядят так:
    # 
    # array([[12., 12., 13., ...,  6.,  6.,  4.],
    #        [11., 13., 15., ...,  7.,  6.,  6.],
    #        [14., 16., 18., ...,  7.,  7.,  6.],
    #        ...,
    #        [24., 21., 25., ..., 31., 31., 24.],
    #        [18., 21., 21., ..., 26., 26., 23.],
    #        [17., 18., 20., ..., 28., 21., 19.]], dtype=float32)
    #
    # "0." - чёрный; "255." - белый

    # Здесь мы получаем двумерное категориальное распределение, 
    # с числом параметров 64 * 64
    # КОТОРОЕ МЫ И ХОТИМ ПРИБЛИЗИТЬ НАШЕЙ МОДЕЛЬЮ
    dist = im / im.sum()

    
    ### СЕМПЛИРОВАНИЕ ИЗ dist
    # pairs перечисляет все возможные пиксели
    # pairs = [(0, 0), (0, 1), ... (63, 62), (63, 63)]
    pairs = list(itertools.product(range(bins), range(bins)))
    # выбираем count пикселей в соответствии с вероятностями из dist
    idxs = np.random.choice(len(pairs), size=count, replace=True, p=dist.reshape(-1))
    samples = np.array([pairs[i] for i in idxs])

    split = int(0.8 * len(samples))
    return dist, samples[:split], samples[split:]

def plot_2d_image_data(train_data, test_data, bins):
    train_dist, test_dist = np.zeros((bins, bins)), np.zeros((bins, bins))
    for i in range(len(train_data)):
        train_dist[train_data[i][0], train_data[i][1]] += 1
    train_dist /= train_dist.sum()

    for i in range(len(test_data)):
        test_dist[test_data[i][0], test_data[i][1]] += 1
    test_dist /= test_dist.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.set_title('Train Data')
    ax1.imshow(train_dist, cmap='gray')
    ax1.axis('off')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x0')

    ax2.set_title('Test Data')
    ax2.imshow(test_dist, cmap='gray')
    ax2.axis('off')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x0')

    plt.show()
    
def draw_contour(density, X, Y, title, n_levels=3):
    plt.figure(figsize=(5, 5))
    density = density.reshape(X.shape)
    levels = np.linspace(np.min(density), np.max(density), n_levels)
    plt.contour(X, Y, density, levels=levels, c='red')
    plt.title(title, fontsize=16)
    plt.show()

def draw_distrib(distrib, title, n_levels=20, x_lim=(-11, 11), y_lim=(-11, 11), dx=0.1, dy=0.1, device='cpu', contour=True, density=True):
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
    densities = torch.exp(distrib.log_prob(torch.tensor(mesh_xs).float().to(device))).detach().cpu().numpy()
    if contour:
        draw_contour(densities, x, y, title='{} contour'.format(title), n_levels=20)
    if density:
        visualize_2d_densities(x, y, densities, title='{} pdf'.format(title))
        
############
# slightly changed function from dgm_utils

def plot_training_curves(
    train_losses, test_losses, logscale_y=False, 
    logscale_x=False, y_lim=None, figsize=None, dpi=None, ewma_span=None, keys=None):
    n_train = len(train_losses[list(train_losses.keys())[0]])
    n_test = len(test_losses[list(train_losses.keys())[0]])
    x_train = np.linspace(0, n_test - 1, n_train)
    x_test = np.arange(n_test)

    plt.figure(figsize=figsize, dpi=dpi)
    for key, value in train_losses.items():
        if not keys or key in keys:
            if ewma_span:
                value = ewma(value, ewma_span)
            plt.plot(x_train, value, label=key + '_train')

    for key, value in test_losses.items():
        if not keys or key in keys:
            if ewma_span:
                value = ewma(value, ewma_span)
            plt.plot(x_test, value, label=key + '_test')

    if logscale_y:
        plt.semilogy()

    if logscale_x:
        plt.semilogx()
    if y_lim:
        plt.ylim(y_lim)

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Loss', fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.grid()
    plt.show()

def visualize_VAE2d_latent_space(model, data, title, device='cuda', figsize=(5, 5), n_levels=20, s=1., offset=0.1):
    data = torch.tensor(data).to(device)
    z, *_ = model(data)
    z = z.detach().cpu()
    x_lim = z[:, 0].min().item() - offset, z[:, 0].max().item() + offset
    y_lim = z[:, 1].min().item() - offset, z[:, 1].max().item() + offset
    dx = (x_lim[1] - x_lim[0])/100.
    dy = (y_lim[1] - y_lim[0])/100.
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
    with torch.no_grad():
        log_probs = model.prior.log_prob(
            torch.tensor(mesh_xs).float().to(device)).detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    density = log_probs.reshape(x.shape)
    levels = np.linspace(np.min(density), np.max(density), n_levels)
    plt.contour(x, y, density, levels=levels, c='red')
    plt.scatter(z[:, 0], z[:, 1], color='black', s=s)
    plt.title(title, fontsize=16)
    plt.show()