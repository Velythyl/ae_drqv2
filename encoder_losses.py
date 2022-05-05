from abc import ABC, abstractmethod

import torch
from torch import nn

import torch.nn.functional as F


class ExtraLoss(nn.Module):
    def __init__(self, own_mixin=1):
        super().__init__()

        self.name = self.__class__.__name__
        self.own_mixin = own_mixin

    @abstractmethod
    def __call__(self, *args) -> (torch.Tensor, dict):
        if self.own_mixin == 0:
            return 0, dict()

        loss = self.loss(*args)
        metrics = {f"{self.name}_loss": loss.item()}

        return loss * self.own_mixin, metrics

    @abstractmethod
    def loss(self, obs, action, reward, discount, next_obs, step):
        raise NotImplementedError()


class ForwardEncoderLoss(ExtraLoss):
    def __init__(self, feature_dim, action_shape, hidden_dim, own_mixin=1):
        super().__init__(own_mixin)
        self.fwd = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, feature_dim))

    def loss(self, obs, action, reward, discount, next_obs, step):
        # obs and next_obs are assumed to have already been trunked

        next_obs = next_obs.detach()

        obs_trunk_act = torch.cat((obs, action), dim=1)
        pred_n_obs_trunk = self.fwd(obs_trunk_act)
        encoder_loss = F.mse_loss(pred_n_obs_trunk, next_obs)

        return encoder_loss

class NoopEncoderLoss(ExtraLoss):
    def __init__(self):
        super().__init__(0)
