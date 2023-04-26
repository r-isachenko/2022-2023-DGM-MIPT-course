import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as TD
from torch import optim

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import matplotlib
import torch.autograd as autograd

from collections import defaultdict
from tqdm.notebook import tqdm

import sys
sys.path.append('../../homeworks')
from dgm_utils.visualize import (
    TICKS_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LABEL_FONT_SIZE,
    TITLE_FONT_SIZE
)


class FullyConnectedMLP(nn.Module):

    def __init__(
        self, 
        input_dim, 
        hiddens, 
        output_dim, 
        activation_gen = lambda : nn.ReLU()
    ):

        assert isinstance(hiddens, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens

        model = []
        prev_h = input_dim
        for h in hiddens:
            model.append(nn.Linear(prev_h, h))
            model.append(activation_gen())
            prev_h = h
        model.append(nn.Linear(hiddens[-1], output_dim))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, self.output_dim)

def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

def batch_jacobian(input, output, create_graph=True, retain_graph=True):
    '''
    :Parameters:
    input : tensor (bs, *shape_inp)
    output: tensor (bs, *shape_oup) , NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, *shape_oup, *shape_inp)
    '''
    def out_permutation():
        n_inp = np.arange(len(input.shape) - 1)
        n_output = np.arange(len(output.shape) - 1)
        return tuple(np.concatenate([n_output + 1, [0,], n_inp + len(n_output) + 1]))

    s_output = torch.sum(output, dim=0) # sum by batch dimension
    batched_grad_outputs = torch.eye(
        np.prod(s_output.shape)).view((-1,) + s_output.shape).to(output)
    # batched_grad_outputs = torch.eye(s_output.size(0)).to(output)
    grad = autograd.grad(
        outputs=s_output, inputs=input,
        grad_outputs=batched_grad_outputs,
        create_graph=create_graph, 
        retain_graph=retain_graph,
        only_inputs=True,
        is_grads_batched=True
    )
    return grad[0].permute(out_permutation())

def visualize_2d_torch_scalar_func(
    func, title, func_transform=lambda x: x,
    val=2., x_lim=None, y_lim=None, dx=0.025, dy=0.025, y_val=0.025,
    figsize=(12, 10), dpi=100, levels=200,
    device=None,
    xlabel=None, ylabel=None):

    if x_lim is None:
        x_lim = (-val, val)
    assert len(x_lim) == 2
    if y_lim is None:
        y_lim = (-val, val)
    assert len(y_lim) == 2

    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = torch.tensor(np.stack([x, y], axis=2).reshape(-1, 2)).to(device).float()
    vals = func_transform(func(mesh_xs)).detach().cpu().numpy()
    vals = vals.reshape(x.shape)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()
    cf = ax.contourf(x, y, vals, levels)
    fig.colorbar(cf, ax=ax)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
