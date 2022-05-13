from abc import ABC, abstractmethod
from functools import cached_property

import torch
from torch import nn

import torch.nn.functional as F


class ExtraLoss(nn.Module):
    def __init__(self, base, for_critic: bool, own_mixin=1):
        super().__init__()

        self.name = self.__class__.__name__

        self.own_mixin = float(own_mixin)
        self.for_critic = for_critic
        self.base = base

    @abstractmethod
    def __call__(self, transition) -> (torch.Tensor, dict):
        base_loss, base_metrics = self.base(transition)

        loss = self._loss(transition)
        metrics = {f"{self.name}_loss": loss.item()}
        if self.own_mixin != 1:
            metrics[f"{self.name}_loss_mixin"] = loss.item() * self.own_mixin

        return loss * self.own_mixin + base_loss, {**metrics, **base_metrics}

    @abstractmethod
    def _loss(self, transition):
        raise NotImplementedError()

class NoopEncoderLoss(ExtraLoss):
    def __init__(self):
        super().__init__(None, None)

    def __call__(self, *args) -> (torch.Tensor, dict):
        return 0, dict()


class ForwardEncoderLoss(ExtraLoss):
    def __init__(self, base, for_critic, feature_dim, action_shape, hidden_dim, own_mixin=1):
        super().__init__(base, for_critic, own_mixin)
        self.fwd = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, feature_dim))

    def _loss(self, transition):
        trunked_obs = transition.trunk_static(self.for_critic)
        detached_trunked_next_obs = transition.trunk_static(self.for_critic)
        action = transition.action

        obs_trunk_act = torch.cat((trunked_obs, action), dim=1)
        pred_n_obs_trunk = self.fwd(obs_trunk_act)
        encoder_loss = F.mse_loss(pred_n_obs_trunk, detached_trunked_next_obs)

        return encoder_loss


class InverseEncoderLoss(ExtraLoss):
    def __init__(self, base, for_critic, feature_dim, action_shape, hidden_dim, own_mixin=1):
        super().__init__(base, for_critic, own_mixin)
        self.bkwd = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_shape[0]))

    def _loss(self, transition):
        trunked_obs = transition.trunk(self.for_critic)
        detached_trunked_next_obs = transition.trunk_static(self.for_critic)
        action = transition.action

        obs_trunk_act = torch.cat((trunked_obs, detached_trunked_next_obs), dim=1)
        pred_actions = self.bkwd(obs_trunk_act)
        encoder_loss = F.mse_loss(pred_actions, action)

        return encoder_loss


class ActionDistanceEncoderLoss(ExtraLoss):
    def __init__(self, base, for_critic, feature_dim, action_shape, hidden_dim, own_mixin=1):
        super().__init__(base, for_critic, own_mixin)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2))

    def _loss(self, transition):
        trunked_obs = transition.trunk(self.for_critic)
        detached_trunked_next_obs = transition.trunk_static(self.for_critic)
        action = transition.action

        obs_trunk_cat = torch.cat((trunked_obs, detached_trunked_next_obs), dim=1)
        pred_vectors = self.projection(obs_trunk_cat)
        pred_angles = pred_vectors @ pred_vectors.T

        action_angles = action @ action.T
        encoder_loss = F.mse_loss(pred_angles, action_angles)

        return encoder_loss

class Transition:
    def __init__(self, actor, critic, obs, action, reward, discount, next_obs, step):
        self.actor = actor
        self.critic = critic

        self.obs = obs
        self.action = action
        self.reward = reward
        self.discount = discount
        self.next_obs = next_obs
        self.step = step

        self._critic_trunked = None
        self._critic_trunked_static = None
        self._actor_trunked = None
        self._actor_trunked_static = None

    def trunk(self, for_critic):
        if for_critic:
            return self.critic_trunk
        else:
            return self.actor_trunk # todo ? change to detached version ? since we never change actor weights from here

    def trunk_static(self, for_critic):
        if for_critic:
            return self.critic_trunk_static
        else:
            return self.actor_trunk_static

    #@cached_property
    @property
    def actor_trunk(self):
        if self._actor_trunked is None:
            self._actor_trunked = self.actor.trunk(self.obs)
        return self._actor_trunked

    #@cached_property
    @property
    def actor_trunk_static(self):
        if self._actor_trunked_static is None:
            self.actor.trunk.requires_grad = False
            self._actor_trunked_static = self.actor.trunk(self.obs)
            self.actor.trunk.requires_grad = True
        return self._actor_trunked_static

    #@cached_property
    @property
    def critic_trunk(self):
        if self._critic_trunked is None:
            self._critic_trunked = self.critic.trunk(self.obs)
        return self._critic_trunked

   # @cached_property
    @property
    def critic_trunk_static(self):
        if self._critic_trunked_static is None:
            self.critic.trunk.requires_grad = False
            self._critic_trunked_static = self.critic.trunk(self.obs)
            self.critic.trunk.requires_grad = True
        return self._critic_trunked_static

class TransitionFactory:
    def __init__(self, actor, critic, base):
        self.actor = actor
        self.critic = critic
        self.base = base

    def __call__(self, obs, action, reward, discount, next_obs, step):
        return self.base(
            Transition(
                self.actor, self.critic,
                obs, action, reward, discount, next_obs, step
            )
        )

class NoopOpt:
    def step(self):
        pass

    def zero_grad(self, set_to_none):
        pass


def build_losses(feature_dim, action_shape, hidden_dim, device, encoder_losses_bools, critic, actor, lr):
    base = NoopEncoderLoss()

    def instantiate(func, own_mixin):
        if own_mixin:
            return func(base, False, feature_dim, action_shape, hidden_dim, own_mixin=own_mixin).to(device)
        else:
            return base

    base = instantiate(ForwardEncoderLoss, encoder_losses_bools.fwd_loss)

    base = instantiate(ActionDistanceEncoderLoss, encoder_losses_bools.action_loss)

    base = instantiate(InverseEncoderLoss, encoder_losses_bools.inverse_loss)


    critic_extra_opt = NoopOpt()
    if isinstance(base, NoopEncoderLoss):
        critic_extra_opt = torch.optim.Adam(base.parameters(), lr=lr)

    return TransitionFactory(actor, critic, base), critic_extra_opt
