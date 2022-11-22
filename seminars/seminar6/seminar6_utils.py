import numpy as np
import torch
import torch.nn as nn

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

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

##############
# Taken from seminar 4

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
