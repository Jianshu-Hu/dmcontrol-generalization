import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.svea import SVEA
from typing import Iterable, List, Tuple
from copy import deepcopy


class WRO(SVEA):
	# weighted random overlay
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.weight_type = args.wro_weight_type

		self.PG = args.projected_grad
		if args.projected_grad:
			self.critic_optimizer = PCGrad(torch.optim.Adam(
				self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
			), reduction='sum')
			self.actor_optimizer = PCGrad(torch.optim.Adam(
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


# Modify PCGrad implemented in
# https://pytorch-optimizers.readthedocs.io/en/latest/_modules/pytorch_optimizer/optimizer/pcgrad.html#PCGrad
class PCGrad():
	"""Gradient Surgery for Multi-Task Learning.

	:param optimizer: OPTIMIZER: optimizer instance.
	:param reduction: str. reduction method.
	"""
	def __init__(self, optimizer, reduction='mean'):
		self.optimizer = optimizer
		self.reduction = reduction
		return

	@torch.no_grad()
	def reset(self):
		self.zero_grad()

	def zero_grad(self):
		return self.optimizer.zero_grad(set_to_none=True)

	def step(self):
		return self.optimizer.step()

	def set_grad(self, grads: List[torch.Tensor]):
		idx: int = 0
		for group in self.optimizer.param_groups:
			for p in group['params']:
				p.grad = grads[idx]
				idx += 1

	def retrieve_grad(self):
		"""Get the gradient of the parameters of the network with specific objective."""

		grad, shape, has_grad = [], [], []
		for group in self.optimizer.param_groups:
			for p in group['params']:
				if p.grad is None:
					shape.append(p.shape)
					grad.append(torch.zeros_like(p, device=p.device))
					has_grad.append(torch.zeros_like(p, device=p.device))
					continue
				shape.append(p.grad.shape)
				grad.append(p.grad.clone())
				has_grad.append(torch.ones_like(p, device=p.device))
		return grad, shape, has_grad

	def pack_grad(self, objectives):
		"""Pack the gradient of the parameters of the network for each objective.

		:param objectives: Iterable[nn.Module]. a list of objectives.
		:return: torch.Tensor. packed gradients.
		"""

		grads, shapes, has_grads = [], [], []
		for objective in objectives:
			self.optimizer.zero_grad(set_to_none=True)
			objective.backward(retain_graph=True)

			grad, shape, has_grad = self.retrieve_grad()

			grads.append(self.flatten_grad(grad))
			has_grads.append(self.flatten_grad(has_grad))
			shapes.append(shape)

		return grads, shapes, has_grads

	def project_conflicting(self, grads: List[torch.Tensor], has_grads: List[torch.Tensor]) -> torch.Tensor:
		"""Project conflicting.
		In our setting, the first task is the main task and th second task is the auxiliary task.

		:param grads: a list of the gradient of the parameters.
		:param has_grads: a list of mask represent whether the parameter has gradient.
		:return: torch.Tensor. merged gradients.
		"""
		shared: torch.Tensor = torch.stack(has_grads).prod(0).bool()
		pc_grad: List[torch.Tensor] = deepcopy(grads)
		# pc_grad[0]: g_main, pc_grad[1]: g_auxiliary
		g_dot = torch.dot(pc_grad[1], pc_grad[0])
		if g_dot < 0:
			pc_grad[1] -= g_dot * pc_grad[0]/(pc_grad[0].norm()**2)
		# for i, g_i in enumerate(pc_grad):
		# 	random.shuffle(grads)
		# 	for g_j in grads:
		# 		g_i_g_j = torch.dot(g_i, g_j)
		# 		if g_i_g_j < 0:
		# 			g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
		merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
		if self.reduction == 'mean':
			merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
		elif self.reduction == 'sum':
			merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
		else:
			exit('invalid reduction method')

		merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
		return merged_grad

	def pc_backward(self, objectives):
		"""Calculate the gradient of the parameters.

		:param objectives: Iterable[nn.Module]. a list of objectives.
		"""

		grads, shapes, has_grads = self.pack_grad(objectives)
		pc_grad = self.project_conflicting(grads, has_grads)
		pc_grad = self.un_flatten_grad(pc_grad, shapes[0])
		self.set_grad(pc_grad)

	def un_flatten_grad(self, grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
		"""Unflatten the gradient"""
		idx: int = 0
		un_flatten_grad: List[torch.Tensor] = []
		for shape in shapes:
			length = np.prod(shape)
			un_flatten_grad.append(grads[idx:idx + length].view(shape).clone())
			idx += length
		return un_flatten_grad

	def flatten_grad(self, grads: List[torch.Tensor]) -> torch.Tensor:
		"""Flatten the gradient."""
		return torch.cat([grad.flatten() for grad in grads])

