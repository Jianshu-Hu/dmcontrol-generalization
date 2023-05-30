import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.svea import SVEA


class DRC(SVEA):
    # distributional random conv with learnable parameters
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.svea_alpha = args.svea_alpha
        self.svea_beta = args.svea_beta

        if args.drc_dist_type == 'normal':
            # parameters of the Gaussian distributions for the random conv kernal
            self.normal_mean = torch.tensor(np.zeros(81), dtype=torch.float).to(self.device)
            self.normal_std = torch.tensor(np.ones(81), dtype=torch.float).to(self.device)
            self.normal_mean.requires_grad = True
            self.normal_std.requires_grad = True
            self.params = [self.normal_mean, self.normal_std]
            # optimizers
            self.dist_optimizer = torch.optim.Adam(self.params, lr=0.0001)
        elif args.drc_dist_type == 'beta':
            # parameters of the Beta distributions for the random conv kernal
            self.beta_con1 = torch.tensor(np.ones(81), dtype=torch.float).to(self.device)
            self.beta_con2 = torch.tensor(np.ones(81), dtype=torch.float).to(self.device)
            self.beta_con1.requires_grad = True
            self.beta_con2.requires_grad = True
            self.params = [self.beta_con1, self.beta_con2]
            # optimizers
            self.dist_optimizer = torch.optim.Adam(self.params, lr=0.0001)
        elif args.drc_dist_type == 'categorical':
            # parameters of the Categorical distributions for the random conv kernal
            n_quantiles = 100
            # 101 numbers from [0,1/100,2/100, .., 100/100]
            self.params = torch.tensor(np.ones((81, n_quantiles+1)), dtype=torch.float).to(self.device)
            # optimizers
            self.dist_optimizer = torch.optim.Adam([self.params], lr=0.0001)
        else:
            raise ValueError('Wrong type of distribution')

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        if self.svea_alpha == self.svea_beta:
            with torch.no_grad():
                obs = utils.cat(obs, augmentations.distributional_random_conv(obs.clone(),
                                      dist_type=self.args.drc_dist_type, params=self.params, with_grad=False))

            action = utils.cat(action, action)
            target_Q = utils.cat(target_Q, target_Q)

            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * \
                          (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
        else:
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * \
                          (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

            with torch.no_grad():
                obs_aug = augmentations.distributional_random_conv(obs.clone(),
                                      dist_type=self.args.drc_dist_type, params=self.params, with_grad=False)
            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.svea_beta * \
                           (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

        if L is not None:
            L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        mu, pi, log_pi, log_std = self.actor(obs, detach=True)
        std = log_std.exp()
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                                ) + log_std.sum(dim=-1)

        # KL loss: KL(A(obs), A(obs_aug))
        # minimize KL loss with the policy
        with torch.no_grad():
            obs_aug = augmentations.distributional_random_conv(obs.clone(),
                                      dist_type=self.args.drc_dist_type, params=self.params, with_grad=False)
        # mu and mu_aug has been squashed (tanh), no further 'torch.tanh' is required.
        mu_aug, _, _, log_std_aug = self.actor(obs_aug, compute_pi=False, compute_log_pi=False, detach=True)
        std_aug = log_std_aug.exp()
        # detach first
        dist = torch.distributions.Normal(mu.detach(), std.detach())
        dist_aug = torch.distributions.Normal(mu_aug, std_aug)

        KL_loss = torch.mean(torch.distributions.kl_divergence(dist, dist_aug))

        actor_loss += self.args.actor_KL_weight * KL_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        # maximize KL loss and minimize the distortion with the data augmentation
        obs_detach = obs.clone().detach()
        if self.args.drc_dist_type == 'categorical':
            obs_aug, log_prob = augmentations.distributional_random_conv(obs_detach,
                                          dist_type=self.args.drc_dist_type, params=self.params, with_grad=True)
        else:
            obs_aug = augmentations.distributional_random_conv(obs_detach,
                                          dist_type=self.args.drc_dist_type, params=self.params, with_grad=True)
        # mu and mu_aug has been squashed (tanh), no further 'torch.tanh' is required.
        mu_aug, _, _, log_std_aug = \
            self.actor(obs_aug, compute_pi=False, compute_log_pi=False, detach=False)
        std_aug = log_std_aug.exp()
        # detach first
        dist_aug = torch.distributions.Normal(mu_aug, std_aug)

        if self.args.drc_dist_type == 'categorical':
            KL_loss = torch.mean(torch.distributions.kl_divergence(dist, dist_aug), -1)
            augmentation_loss = -KL_loss + torch.mean(torch.square((obs - obs_aug) / 255.0), dim=(3, 2, 1))
            augmentation_loss = torch.mean(log_prob*augmentation_loss)
        else:
            KL_loss = torch.mean(torch.distributions.kl_divergence(dist, dist_aug))
            augmentation_loss = -KL_loss + torch.mean(torch.square((obs-obs_aug)/255.0), dim=(3, 2, 1, 0))

        if L is not None:
            L.log('train/aux_loss', augmentation_loss, step)

        self.dist_optimizer.zero_grad()
        augmentation_loss.backward()
        self.dist_optimizer.step()
