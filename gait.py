import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn


class Gait(nn.Module):
    def __init__(self, nb_gaussians, action_shape):
        super().__init__()
        self.mixture_dim = (nb_gaussians, action_shape[0])

        # same init as
        # https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
        self.mu_matrix = nn.Parameter(torch.normal(0,1,self.mixture_dim), requires_grad=True)
        self.sigma_matrix = nn.Parameter(-torch.ones(self.mixture_dim), requires_grad=True)

        self.weights = nn.Parameter(torch.normal(0,1,self.mixture_dim), requires_grad=True)
        # initial period is to have a ~25 frame period. Learnable parameter.
        self.period_b = nn.Parameter(torch.tensor(0.25), requires_grad=True)

    @property
    def period(self):
        return torch.floor((np.pi * 2) / self.period_b) # .clone().detach().item()

    def frame2percent(self, frame_nb: torch.Tensor):    # only to be used as reference for the actor to know which part of the cycle we're in
        return frame_nb / self.period

    def cyclic_gaussian_mixture(self, frame_numbers):
        # https://studywolf.wordpress.com/tag/rhythmic-dynamic-movement-primitives/

        x_mu = frame_numbers - self.mu_matrix
        cos_x_mu = torch.cos(x_mu)

        scaled_x_mu_pow = torch.mul(self.sigma_matrix, cos_x_mu) - 1

        gaussian = torch.exp(scaled_x_mu_pow)
        return gaussian

    def forward(self, frame_nb):
        frame_nb = self.frame2percent(frame_nb).unsqueeze(-1)
        frame_nb_batch = frame_nb.unsqueeze(-1).expand(frame_nb.shape[0], *self.mixture_dim)

        DEBUG = frame_nb_batch.detach().numpy()

        mixed = self.cyclic_gaussian_mixture(frame_nb_batch)
        DEBUG = mixed.detach().numpy()

        #mixed_weighted = torch.mul(mixed, self.weights)

        #print(mixed.shape)

        #mixed = torch.sum(mixed, dim=1)
        #mixed_weighted = torch.sum(mixed_weighted, dim=1)

       # activations = mixed_weighted / mixed

        #print(activations.shape)

        return torch.sum(mixed, dim=1)

if __name__ == "__main__":
    FRAMES = 50
    gait = Gait(100, [1])
    opt = torch.optim.Adam(gait.parameters())

    x = torch.arange(0, FRAMES, requires_grad=False)

    y = torch.normal(0.2, 1, (FRAMES,), requires_grad=False)
    z = torch.normal(0.2, 1, (FRAMES,), requires_grad=False)
    y = y + z

    plt.plot(x, y)
    plt.plot(gait(x).detach().cpu().numpy())
    plt.show()

    mse = torch.nn.MSELoss()
    for i in range(100000):
        y_pred = gait(x)
        loss = mse(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.detach().item())

    plt.plot(x, y)
    plt.plot(y_pred.detach().cpu().numpy())
    plt.show()
    pass