import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.svea import SVEA


class DRO(SVEA):
    # distributional random overlay
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        dataset_imgs_num = augmentations.initialize_dataset(batch_size=args.batch_size,
                                    image_size=args.image_size, dataset='places365_standard')
        self.params = torch.tensor(np.zeros(dataset_imgs_num), dtype=torch.float).to(self.device)
        self.params.requires_grad = True
        # optimizers
        self.dist_optimizer = torch.optim.Adam([self.params], lr=0.0001)

        self.critic_optimizer = m.PCGrad(torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        ), reduction='sum')
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.prob_loss_weight = 0.1

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = self.svea_alpha * \
                      (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

        obs_aug, log_prob = augmentations.distributional_random_overlay(obs.clone(), params=self.params)
        current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
        # critic_loss += self.svea_beta * \
        #                (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))
        # if L is not None:
        #     L.log('train_critic/loss', critic_loss, step)
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        critic_loss_aug = self.svea_beta * \
                       (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

        if L is not None:
            L.log('train_critic/loss', critic_loss + critic_loss_aug, step)
            L.log('train_critic/original_loss', critic_loss, step)
            L.log('train_critic/aug_loss', critic_loss_aug, step)

        loss = [critic_loss, critic_loss_aug]
        # self.critic_optimizer.zero_grad()
        self.critic_optimizer.pc_backward(loss)
        self.critic_optimizer.step()

        # maximize the Q-diff
        with torch.no_grad():
            original_Q = torch.min(current_Q1, current_Q2)
            Q_diff = torch.square(current_Q1_aug - original_Q)+torch.square(current_Q2_aug-original_Q)
        prob_loss = torch.mean(-log_prob*Q_diff)

        if L is not None:
            L.log('train_critic/prob_loss', prob_loss, step)

        self.dist_optimizer.zero_grad()
        prob_loss.backward()
        self.dist_optimizer.step()

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

        # # KL loss: KL(A(obs), A(obs_aug))
        # # maximize KL loss
        # obs_aug, log_prob = augmentations.distributional_random_overlay(obs.clone(), params=self.params)
        # with torch.no_grad():
        #     # mu and mu_aug has been squashed (tanh), no further 'torch.tanh' is required.
        #     mu_aug, _, _, log_std_aug = self.actor(obs_aug, compute_pi=False, compute_log_pi=False, detach=True)
        #     std_aug = log_std_aug.exp()
        #     dist = torch.distributions.Normal(mu, std)
        #     dist_aug = torch.distributions.Normal(mu_aug, std_aug)
        #
        #     KL_diff = torch.mean(torch.distributions.kl_divergence(dist, dist_aug), -1, keepdim=True)
        #
        # prob_loss = torch.mean(-log_prob*KL_diff)
        #
        # if L is not None:
        #     L.log('train_actor/prob_loss', prob_loss, step)
        #
        # self.dist_optimizer.zero_grad()
        # prob_loss.backward()
        # self.dist_optimizer.step()

