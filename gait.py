import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn


class Gait(nn.Module):
    def __init__(self, nb_gaussians, action_shape):
        super().__init__()
        self.mixture_dim = (action_shape[0], nb_gaussians)

        # initial period is to have a ~25 frame period. Learnable parameter.
        self.period_b = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # same init as
        mu_matrix_init = np.linspace(0, 2*np.pi, nb_gaussians + 1)
        mu_matrix_init = mu_matrix_init[0:-1]
        mu_matrix_init = torch.tensor(action_shape[0]*[mu_matrix_init])
        # https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
        self.mu_matrix = mu_matrix_init.float() #nn.Parameter(mu_matrix_init.float(), requires_grad=False)
        self.sigma_matrix = nn.Parameter(-torch.ones(self.mixture_dim), requires_grad=True)

        self.weights = nn.Parameter(torch.normal(0,1,self.mixture_dim), requires_grad=True)

    def period(self):
        return (np.pi * 2) / self.period_b # .clone().detach().item()

    def frame2percent(self, frame_nb: torch.Tensor):    # only to be used as reference for the actor to know which part of the cycle we're in
        return frame_nb / self.period

    def cyclic_gaussian_mixture(self, frame_numbers):
        # https://studywolf.wordpress.com/tag/rhythmic-dynamic-movement-primitives/

        x_mu = frame_numbers - self.mu_matrix

        DEBUG = x_mu.squeeze().detach().numpy()

        cos_x_mu = torch.cos(self.period_b * x_mu)

        scaled_x_mu_pow = torch.mul(self.sigma_matrix, cos_x_mu) - 1

        gaussian = torch.exp(scaled_x_mu_pow)
        return gaussian

    def forward(self, frame_nb):
        #frame_nb = frame_nb / self.period_b

       # frame_nb = self.frame2percent(frame_nb) * np.pi * 2

        #frame_nb_batch = frame_nb.repeat(frame_nb, *self.mixture_dim)

        frame_nb_batch = frame_nb.unsqueeze(-1).unsqueeze(-1).expand(frame_nb.shape[0], *self.mixture_dim)

        DEBUG = frame_nb_batch.squeeze().detach().numpy()

        mixed = self.cyclic_gaussian_mixture(frame_nb_batch)
        DEBUG = mixed.squeeze().detach().numpy()

        mixed_weighted = torch.mul(mixed, self.weights)

        #print(mixed.shape)

        mixed = torch.sum(mixed, dim=-1)
        mixed_weighted = torch.sum(mixed_weighted, dim=-1)

        activations = mixed_weighted #mixed_weighted / mixed

        DEBUG2 = activations.squeeze().detach().numpy()

        #print(activations.shape)

        #final = torch.sum(mixed, dim=-1)

        #DEBUG = final.squeeze().detach().numpy()

        #return final
        return activations

if __name__ == "__main__":
    FRAMES = 50
    gait = Gait(500, [1])
    opt = torch.optim.Adam(gait.parameters())

    x = torch.arange(0, FRAMES, requires_grad=False)

    y = torch.normal(0.2, 1, (FRAMES,1), requires_grad=False)
    y2 = torch.normal(0.2, 1, (FRAMES,1), requires_grad=False)
    y = y + y2

    target = y

    plt.plot(x, y)
    plt.plot(gait(x).detach().cpu().numpy())
    plt.show()

    print(gait.period)

    mse = torch.nn.MSELoss()
    for i in range(10000):
        y_z_pred = gait(x)

        DEBUG = y_z_pred.squeeze().detach().numpy()

        loss = mse(y_z_pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.detach().item())

    print(gait.period_b)

    plt.plot(x, y)
    plt.plot(y_z_pred.detach().cpu().numpy())
    plt.show()
    pass