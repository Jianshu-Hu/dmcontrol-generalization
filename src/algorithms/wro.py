import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.svea import SVEA


class WRO(SVEA):
	# weighted random overlay
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.weight_type = args.wro_weight_type

		self.PG = args.projected_grad
		if args.projected_grad:
			self.critic_optimizer = m.PCGrad(torch.optim.Adam(
				self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
			), reduction='sum')
			self.actor_optimizer = m.PCGrad(torch.optim.Adam(
				self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
			), reduction='mean')

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

		if self.args.complex_DA == 'cut_random_overlay':
			data_aug = augmentations.cut_random_overlay
		elif self.args.complex_DA == 'random_overlay':
			data_aug = augmentations.random_overlay
		else:
			raise ValueError('Undefined data augmentation')

		obs_aug = data_aug(obs.clone())
		current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)

		if self.weight_type == 'q_diff' or self.weight_type == 'q_diff_kl_diff':
			# set the weight for each sample according to the q-diff
			with torch.no_grad():
				original_Q = torch.min(current_Q1, current_Q2)
				weight1 = F.softmax(torch.square(current_Q1_aug-original_Q))
				weight2 = F.softmax(torch.square(current_Q2_aug-original_Q))
		elif self.weight_type == 'q_diff_div_image_diff':
			# set the weight for each sample according to the q-diff and image diff
			with torch.no_grad():
				original_Q = torch.min(current_Q1, current_Q2)
				image_diff = torch.mean(torch.square((obs-obs_aug)/255.0), dim=(3, 2, 1)).reshape(original_Q.size(0), 1)
				weight1 = F.softmax(torch.square(current_Q1_aug-original_Q)/image_diff)
				weight2 = F.softmax(torch.square(current_Q2_aug-original_Q)/image_diff)

		if self.PG:
			critic_loss_aug = self.svea_beta * \
						(self.weighted_mse(current_Q1_aug, target_Q, weight1)
							+ self.weighted_mse(current_Q2_aug, target_Q, weight2))

			if L is not None:
				L.log('train_critic/loss', critic_loss+critic_loss_aug, step)
				L.log('train_critic/original_loss', critic_loss, step)
				L.log('train_critic/aug_loss', critic_loss_aug, step)

			loss = [critic_loss, critic_loss_aug]
			# self.critic_optimizer.zero_grad()
			self.critic_optimizer.pc_backward(loss)
			self.critic_optimizer.step()
		else:
			critic_loss += self.svea_beta * \
						(self.weighted_mse(current_Q1_aug, target_Q, weight1)
						+ self.weighted_mse(current_Q2_aug, target_Q, weight2))

			if L is not None:
				L.log('train_critic/loss', critic_loss, step)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

	def weighted_mse(self, x, y, weights):
		weighted_mse_loss = torch.sum(weights*torch.square(x-y))
		return weighted_mse_loss

	def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
		mu, pi, log_pi, log_std = self.actor(obs, detach=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if self.PG:
			if self.args.complex_DA == 'cut_random_overlay':
				data_aug = augmentations.cut_random_overlay
			elif self.args.complex_DA == 'random_overlay':
				data_aug = augmentations.random_overlay
			else:
				raise ValueError('Undefined data augmentation')

			obs_aug = data_aug(obs.clone())
			mu_aug, pi_aug, log_pi_aug, log_std_aug = self.actor(obs_aug, detach=True)
			actor_Q1_aug, actor_Q2_aug = self.critic(obs_aug, pi_aug, detach=True)
			actor_Q_aug = torch.min(actor_Q1_aug, actor_Q2_aug)

			if self.weight_type == 'q_diff_kl_diff':
				# weight the augmented samples with the KL divergence
				with torch.no_grad():
					dist = torch.distributions.Normal(mu, log_std.exp())
					dist_aug = torch.distributions.Normal(mu_aug, log_std_aug.exp())
					KL_divergence = torch.mean(torch.distributions.kl_divergence(dist, dist_aug), -1, keepdim=True)
					weights = F.softmax(KL_divergence)
				actor_loss_aug = torch.sum(weights*(self.alpha.detach() * log_pi_aug - actor_Q_aug))
			else:
				actor_loss_aug = (self.alpha.detach() * log_pi_aug - actor_Q_aug).mean()

			if L is not None:
				L.log('train_actor/loss', (actor_loss + actor_loss_aug) / 2, step)
				L.log('train_actor/original_loss', (actor_loss) / 2, step)
				L.log('train_actor/aug_loss', (actor_loss_aug) / 2, step)

			loss = [actor_loss, actor_loss_aug]
			# self.actor_optimizer.zero_grad()
			self.actor_optimizer.pc_backward(loss)
			self.actor_optimizer.step()
		else:
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


