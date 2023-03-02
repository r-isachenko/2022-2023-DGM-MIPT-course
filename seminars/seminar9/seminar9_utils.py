import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt

class FullyConnectedMLP(nn.Module):

    def __init__(self, input_dim, hiddens, output_dim):
        assert isinstance(hiddens, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens

        model = []
        prev_h = input_dim
        for h in hiddens:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.append(nn.Linear(hiddens[-1], output_dim))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, self.output_dim)
    
###############
# Taken from seminar 6

def batch_jacobian(input, output, create_graph=True, retain_graph=True):
    '''
    :Parameters:
    input : tensor (bs, D)
    output: tensor (bs, d) , NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, d, D)
    '''
    s_output = torch.sum(output, dim=0) # (d,)
    batched_grad_outputs = torch.eye(s_output.size(0)).to(output) #(d, d)
    grad = autograd.grad(
        outputs=s_output, inputs=input,
        grad_outputs=batched_grad_outputs,
        create_graph=create_graph, 
        retain_graph=retain_graph,
        only_inputs=True,
        is_grads_batched=True
    ) # (d, bs, D) 
    return grad[0].permute(1, 0, 2)

###########
# current seminar functions

def make_inference(generator, critic, n_samples=5000, compute_grad_norms=True):
    generator.eval()
    critic.eval()
    xs = np.linspace(-3.0, 3.0, 1000 + 1)
    xg, yg = np.meshgrid(xs, xs)
    grid = np.concatenate((xg.reshape(-1, 1), yg.reshape(-1, 1)), axis=-1)

    tsr_grid = torch.FloatTensor(grid).to(next(iter(generator.parameters())))
    with torch.no_grad():
        samples = generator.sample(n_samples).cpu().detach().numpy()
        critic_output = critic(tsr_grid).cpu().detach().numpy()

    if compute_grad_norms:
        tsr_grid.requires_grad_() # (grid_size, 2)
        _critic_output = critic(tsr_grid) # (grid_size, 1)
        assert len(critic_output.shape) == 2
        grads = batch_jacobian(
            tsr_grid, _critic_output, 
            create_graph=False, retain_graph=False).squeeze(1) # (grid_size, 2)
        critic_grad_norms = torch.norm(grads, dim=-1).detach().cpu().numpy().reshape((1000 + 1, 1000 + 1))

    critic_output = np.prod(critic_output, axis=-1).reshape((1000 + 1, 1000 + 1))
    if compute_grad_norms:
        return samples, grid, critic_output, critic_grad_norms
    return samples, grid, critic_output

def visualize_GAN_output(
    generated_samples, real_samples, grid, 
    critic_output, critic_grad_norms, npts=100 + 1):

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # plt.figure(figsize=(6, 6))
    # plt.gca().set_aspect("equal")

    npts = critic_output.shape[0]
    cnt = ax[0].contourf(
        grid[:, 0].reshape((npts, npts)), grid[:, 1].reshape((npts, npts)), critic_output,
        levels=25, cmap="cividis"
    )
    ax[0].scatter(generated_samples[:, 0], generated_samples[:, 1], marker=".", color="red", s=0.5)
    ax[0].scatter(real_samples[:, 0], real_samples[:, 1], marker="x", color="blue", s=0.5)
    ax[0].set_title('Critic/discriminator outputs')
    fig.colorbar(cnt, ax=ax[0])
    cnt = ax[1].contourf(
        grid[:, 0].reshape((npts, npts)), grid[:, 1].reshape((npts, npts)), critic_grad_norms,
        levels=25, cmap="cividis"
    )
    ax[1].set_title('Norms of critic/discriminator gradients')
    fig.colorbar(cnt, ax=ax[1])
    plt.show()
