# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from activations import string2activation
from losses.losses import NoopEncoderLoss, NoopOpt, build_losses


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, actor_activation, gait):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        ActorActivation = string2activation(actor_activation, hidden_dim)

        self.policy = nn.Sequential(nn.Linear(feature_dim if not gait else (feature_dim + 1), hidden_dim),
                                    ActorActivation(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    ActorActivation(),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.mu_activation = nn.Tanh() if actor_activation != "snake" else ActorActivation()

        self.gait = gait

        self.apply(utils.weight_init)

    def forward(self, obs, std, frame_nb):
        h = self.trunk(obs)

        if self.gait:
            phi = self.gait.phi(frame_nb)
            h = torch.cat((h, phi), dim=1)  # todo dim

        mu = self.policy(h)
        mu = self.mu_activation(mu)

        mu = mu
        if self.gait:
            mu = mu + self.gait(phi)

        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class DistributionAddition:
    def __init__(self, distr, add):
        self.distr = distr
        self.add = add

    def sample(self, clip):
        return self.distr.sample(clip) + self.add

    @property
    def mean(self):
        return self.distr.mean + self.add

    def log_prob(self, x):
        return self.distr.log_prob(x) + self.add


class Gait(nn.Module):
    def __init__(self, nb_gaussians, action_shape):
        super().__init__()
        self.mixture_dim = (nb_gaussians, action_shape[0])

        # same init as
        # https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
        self.mu_matrix = nn.Parameter(torch.normal(0,1,self.mixture_dim), requires_grad=True)
        self.sigma_matrix = nn.Parameter(-torch.zeros(self.mixture_dim), requires_grad=True)

        self.weights = nn.Parameter(torch.normal(0,1,self.mixture_dim), requires_grad=True)
        # initial period is to have a ~25 frame period. Learnable parameter.
        self.period = nn.Parameter(torch.tensor([[0.25]]), requires_grad=True)

    def phi(self, frame_nb):
        return torch.cos(self.period * torch.tensor(frame_nb))

    def gaussian_mixture(self, phi):
        x_mu = phi - self.mu_matrix
        x_mu_pow = x_mu.pow(2)
        scaled_x_mu_pow = torch.mul(self.sigma_matrix, x_mu_pow)
        gaussian = torch.exp(scaled_x_mu_pow)
        return gaussian

    def forward(self, phi):
        # reshape phis to compute batches
        phi_batch = phi.unsqueeze(-1).expand(phi.shape[0], *self.mixture_dim)

        #print(phi_batch.shape)

        mixed = self.gaussian_mixture(phi_batch)

        mixed_weighted = torch.mul(mixed, self.weights)

        #print(mixed.shape)

        mixed = torch.sum(mixed, dim=1)
        mixed_weighted = torch.sum(mixed_weighted, dim=1)

        activations = mixed_weighted / mixed

        #print(activations.shape)

        return activations


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, already_trunked=False):
        h = obs if already_trunked else self.trunk(obs)

        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, encoder_losses, actor_activation, with_gait):

        print(encoder_losses)  # todo rm this

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.gait = Gait(100, action_shape) if with_gait else False
        self.with_gait = with_gait
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, actor_activation, self.gait).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gait_opt = torch.optim.Adam(self.gait.parameters(), lr=lr) if isinstance(self.gait, Gait) else NoopOpt()

        # extra_losses
        # extra losses need access to critic's trunk but the weights belong to critic, so we wrap it in a lambda
        self.critic_extra, self.critic_extra_opt = build_losses(feature_dim, action_shape, hidden_dim, device,
                                                                encoder_losses, self.critic, self.actor, lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, frame_nb, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev, frame_nb)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def encoder_loss(self, obs, action, reward, next_obs, step):
        pass

    def update_critic(self, obs, action, reward, discount, next_obs, step, frame_nb):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev, frame_nb)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, already_trunked=False)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        extra_loss, extra_metrics = self.critic_extra(obs, action, reward, discount, next_obs, step)
        metrics = {**metrics, **extra_metrics}  # merge

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.critic_extra_opt.zero_grad(set_to_none=True)
        (critic_loss + extra_loss).backward()
        """
        for param in self.encoder.parameters():
            temp = torch.zeros(param.grad.shape)
            temp[param.grad != 0] += 1
            print(torch.any(temp >= 1))
        for param in self.critic.parameters():
            temp = torch.zeros(param.grad.shape)
            temp[param.grad != 0] += 1
            print(torch.any(temp >= 1))
        exit()"""
        self.critic_opt.step()
        self.encoder_opt.step()
        self.critic_extra_opt.step()

        return metrics

    def update_actor(self, obs, step, frame_nb):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev, frame_nb)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            if self.with_gait:
                metrics['actor_gait_period'] = self.actor.gait.period.clone().detach().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, frame_nb = utils.to_torch(
            batch, self.device)

        # augment
        aug_obs = self.aug(obs.float())
        aug_next_obs = self.aug(next_obs.float())
        # encode
        aug_obs = self.encoder(aug_obs)
        with torch.no_grad():
            aug_next_obs = self.encoder(aug_next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(aug_obs, action, reward, discount, aug_next_obs, step, frame_nb))

        # update actor
        metrics.update(self.update_actor(aug_obs.detach(), step, frame_nb))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
