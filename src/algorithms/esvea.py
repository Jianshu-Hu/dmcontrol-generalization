import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
import algorithms.equivariant_module as equim
from algorithms.esac import ESAC


class ESVEA(ESAC):
	# equivariant SVEA
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta

		self.args = args
		self.export_timesteps = args.export_timesteps

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			if self.args.complex_DA == 'random_conv':
				obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
			if self.args.complex_DA == 'random_overlay':
				obs = utils.cat(obs, augmentations.random_overlay(obs.clone()))
			
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			if self.args.complex_DA == 'random_conv':
				obs_aug = augmentations.random_conv(obs.clone())
			if self.args.complex_DA == 'random_overlay':
				obs_aug = augmentations.random_overlay(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		if self.export_timesteps == 0 or step == self.export_timesteps:
			# 0 means exporting the encoder at the beginning
			# otherwise, export the encoder at specified step
			shared_cnn = equim.ExportEncoder(self.critic.encoder.shared_cnn)
			head_cnn = self.critic.encoder.head_cnn
			actor_encoder = m.Encoder(
				shared_cnn,
				head_cnn,
				self.actor.encoder.projection
			)
			critic_encoder = m.Encoder(
				shared_cnn,
				head_cnn,
				self.critic.encoder.projection
			)

			self.actor = m.Actor(actor_encoder, self.action_shape, self.args.hidden_dim,
								self.args.actor_log_std_min, self.args.actor_log_std_max).to(self.device)
			self.critic = m.Critic(critic_encoder, self.action_shape, self.args.hidden_dim).to(self.device)
			self.critic_target = deepcopy(self.critic)

			self.actor_optimizer = torch.optim.Adam(
				self.actor.parameters(), lr=self.args.actor_lr, betas=(self.args.actor_beta, 0.999)
			)
			self.critic_optimizer = torch.optim.Adam(
				self.critic.parameters(), lr=self.args.critic_lr, betas=(self.args.critic_beta, 0.999)
			)

			self.train()
			self.critic_target.train()
			print('------Remove the constraints on the encoder------')
			self.export_steps = -1
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
