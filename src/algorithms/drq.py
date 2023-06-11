import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC


class DrQ(SAC): # [K=2, M=2]
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		# data aug type: shift or rot
		self.data_aug_type = args.data_aug_type
		self.degrees = args.degrees

	def update_critic(self, obs_aug_1, obs_aug_2, action, reward,
					next_obs_aug_1, next_obs_aug_2, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs_aug_1)
			target_Q1, target_Q2 = self.critic_target(next_obs_aug_1, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q_aug_1 = reward + (not_done * self.discount * target_V)

			_, policy_action, log_pi, _ = self.actor(next_obs_aug_2)
			target_Q1, target_Q2 = self.critic_target(next_obs_aug_2, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q_aug_2 = reward + (not_done * self.discount * target_V)

			target_Q = (target_Q_aug_1+target_Q_aug_2)/2

		current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_aug_1, action)
		critic_loss = F.mse_loss(current_Q1_aug_1, target_Q) + F.mse_loss(current_Q2_aug_1, target_Q)

		current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_aug_2, action)
		critic_loss += F.mse_loss(current_Q1_aug_2, target_Q) + F.mse_loss(current_Q2_aug_2, target_Q)

		critic_loss = critic_loss/2
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		obs_aug_1, obs_aug_2, action, reward, next_obs_aug_1, next_obs_aug_2, not_done =\
			replay_buffer.sample_drq(degrees=self.degrees, data_aug_type=self.data_aug_type)

		self.update_critic(obs_aug_1, obs_aug_2, action, reward, next_obs_aug_1, next_obs_aug_2, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs_aug_1, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
